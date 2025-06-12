# -*- coding: utf-8 -*-
# file: grpo_plugin.py
# date: 2025-04-28


import pdb
import os
import sys
import pdb
import json
import asyncio
import nest_asyncio            
from pydantic import BaseModel
from typing import Dict, List, Optional, Annotated, Literal, Callable, Type
from langchain_core.messages import AIMessage
from langchain_core.messages import AnyMessage
from langchain_core.messages import HumanMessage
from langchain_core.messages import SystemMessage
from langchain_core.language_models import BaseChatModel
from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from langgraph.pregel.io import AddableValuesDict
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model


SYS_PROMPT_FACTUALITY: str = \
"""
You're a document verification expert. 

Check whether all the information in response can be:
* Fully Supported and contained by Evidence.
* Totally accurate according to the given input and instruction.

<given_input>
__IN_TEXT__
</given_input>

<instruction>
__INSTRUCTION__
</instruction>

<response>
__RESPONSE__
</response>

You must output only a JSON which contains following keys in order:
* rationale: A brief explanation for the assigned label.
* label: One of 'supported', 'unsupported'
The output have to be enclosed with <answer> and </answer>.
""".strip("\n")


TEST_IN_TEXT: str = \
"""
Patient ID: 12345

Time 1: April 28, 2025, 09:15 AM
Heart Rate: 85 bpm (Normal)
Blood Pressure: 120/80 mmHg (Normal)
Respiratory Rate: 16 breaths per minute (Normal)
Temperature: 98.6°F (37.0°C) (Normal)
Oxygen Saturation: 98% on room air (Normal)

The patient appears well and is in no acute distress. Vital signs are within normal ranges.

Time 2: April 29, 2025, 02:45 PM
Heart Rate: 95 bpm (Slightly elevated)
Blood Pressure: 130/85 mmHg (Prehypertensive range)
Respiratory Rate: 20 breaths per minute (Elevated)
Temperature: 100.4°F (38.0°C) (Low-grade fever)
Oxygen Saturation: 95% on room air (Slightly reduced)

The patient reports feeling fatigued and experiencing mild headache. 
""".strip("\n")


TEST_INSTRUCTION: str = \
"""
Extract latest heart rate.
""".strip("\n")


TEST_GEN_TEXT: str = \
"""
Heart Rate: 85 bpm (Normal)     
""".strip("\n")


class OverallState(BaseModel):
    llms: Dict[str, BaseChatModel] = {}
    msgs_factuality: List[AnyMessage] = []
    msgs_eligibility: List[AnyMessage] = []
    instruction: Optional[str] = None
    in_text: Optional[str] = None
    gen_text: Optional[str] = None
    score_factuality: Optional[float] = None


class Config(BaseModel):
    model_factuality: Optional[str] = None


def text_tag_extract(text: str, start: str, end: str) -> Optional[str]:
    try:
        start_idx: int = text.index(start) + len(start)
        end_idx: int = text.index(end)
        return text[start_idx:end_idx]
    except Exception as e:
        return None


async def agent_factuality(
    state: OverallState, 
    config: Config
) -> Dict:
    cus_config: Dict = config["configurable"]
    llm: str = state.llms[cus_config["model_factuality"]]
    msgs: List[AnyMessage] = state.msgs_factuality
    if msgs == []:
        instruction: str = state.instruction
        in_text: str = state.in_text
        gen_text: str = state.gen_text
        sys_prompt: str = SYS_PROMPT_FACTUALITY
        sys_prompt = sys_prompt\
            .replace("__IN_TEXT__", in_text)\
            .replace("__INSTRUCTION__", instruction)\
            .replace("__RESPONSE__", gen_text)
        msgs.append(HumanMessage(content=sys_prompt))
    resp: AIMessage = await llm.ainvoke(msgs)
    msgs.append(resp)
    
    out: Dict = {
        "msgs_factuality": msgs, 
        "score_factuality": None
    }
    result_raw: str = text_tag_extract(resp.content, "<answer>", "</answer>")
    try:
        result: Dict = json.loads(result_raw)
        if result["label"].lower() == "supported":
            out["score_factuality"] = 1.0
        else:
            out["score_factuality"] = 0.0
    except Exception as e:
         out["score_factuality"] = 0.5
    return out


def agents_init(
    state_t: Type, 
    config_t: Type,
    agent_factuality: Callable 
) -> CompiledStateGraph:
    builder: StateGraph = StateGraph(state_t, config_schema=config_t)
    builder.add_node("agent_factuality", agent_factuality)
    builder.add_edge(START, "agent_factuality")
    builder.add_edge("agent_factuality", END)
    graph: CompiledStateGraph = builder.compile()          
    return graph


def llms_init(configs: List[Dict]) -> Dict[str, BaseChatModel]:
    return {
        x["model"]: init_chat_model(**x) for x in configs
    }


def chatml_to_text(chatml: List[Dict]) -> str:
    text: str = ""
    if len(chatml) == 1:
        text = chatml[0]["content"]
    for msg in chatml:
        role: str = msg["role"]
        content: str = msg["content"]
        text += "\n<%s>" % role
        text += content
        text += "\n</%s>" % role
    return text


class Reward:
    def __init__(self):
        self.llms: Optionanl[Dict[str, BaseChatModel]] = None
        self.agents: Optional[CompiledStateGraph] = None
        self.llm_factuality: Optional[str] = None
        self.instruction_idx: Optional[str] = None
        self.in_text_idx: Optional[str] = None
    
    @classmethod
    def new_with_configs(cls, configs: Dict):
        out = cls()
        llms_configs: List[Dict] = configs["llms"]

        out.llms = llms_init(llms_configs)
        out.agents = agents_init(
            state_t=OverallState, 
            config_t=Config,
            agent_factuality=agent_factuality
        )
        out.llm_factuality = configs["llm_factuality"]
        out.instruction_idx = configs["msg_idx_instruction"]
        out.in_text_idx = configs["msg_idx_in_text"]
        return out

    async def arun(self, 
        prompts: List[List[Dict]], 
        completions: List[List[Dict]]
    ) -> List[float]:
        assert(len(prompts) == len(completions))
        out: List[float] = []
        agents_conf: Dict = {
            "configurable": Config(
                model_factuality=self.llm_factuality
            ).dict()
        }
        for i in range(len(prompts)):
            in_text: str = prompts[i][self.in_text_idx]["content"]
            instruction: str = prompts[i][self.instruction_idx]["content"]
            answer: str = completions[i][0]["content"]
            inputs: Dict = {
                "llms": self.llms, 
                "instruction": instruction,
                "in_text": in_text,
                "gen_text": answer
            }
            result: OverallState = (
                await self.agents.ainvoke(inputs, config=agents_conf)
            )
            out.append(result["score_factuality"])
        return out
    
    def run(self, 
        prompts: List[List[Dict]],                
        completions: List[List[Dict]], 
        **kwargs
    ) -> List[float]:
        nest_asyncio.apply()
        loop = asyncio.get_running_loop()
        coro = self.arun(prompts, completions)
        if loop.is_running():
            return loop.run_until_complete(coro)
        else:
            return asyncio.run(coro)


async def test(configs: Dict) -> None:
    model_factuality: str = "unsloth/gemma-3-27b-it-bnb-4bit" 
    rm: Reward = Reward.new_with_configs(configs)
    prompts: List[Dict] = [
        {"role": "system", "content": TEST_INSTRUCTION},
        {"role": "user", "content": TEST_IN_TEXT}
    ]
    completions: List[Dict] = [
        {"role": "assistant", "content": TEST_GEN_TEXT}
    ]
    rewards = rm.run([prompts], [completions])
    assert(rewards[0] == 0.0)
