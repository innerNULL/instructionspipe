# -*- coding: utf-8 -*-
# file: v0.py
# date: 2025-05-24


import os
import sys
import pdb
import json
import asyncio
import io
import tempfile
import subprocess
import uuid
import uvicorn
import logging
from pydantic import BaseModel, RootModel
from typing import Dict, List, Optional, Annotated, Literal, Callable, Union, Type, Any, Tuple
from langchain_core.messages import AIMessage
from langchain_core.messages import AnyMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_core.prompts import HumanMessagePromptTemplate
from langchain_core.messages import SystemMessage
from langchain_core.language_models import BaseChatModel
from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from langgraph.pregel.io import AddableValuesDict
from langgraph.types import Command, Send
from langchain_core.caches import InMemoryCache
from langchain_core.messages.utils import convert_to_openai_messages
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model
from langchain_core.globals import set_llm_cache

from instructionspipe.impl.agentic_mr import Instruction
from instructionspipe.impl.agentic_mr import AgentMeta
from instructionspipe.impl.agentic_mr import State
from instructionspipe.impl.agentic_mr import RouterState
from instructionspipe.impl.agentic_mr import agent_supervisor
from instructionspipe.impl.agentic_mr import edge_map
from instructionspipe.impl.agentic_mr import agents_build


def langchain_init(configs: Dict) -> None:
    os.environ["LANGSMITH_TRACING"] = str(configs["langsmith_tracing"]).lower()
    os.environ["LANGSMITH_ENDPOINT"] = configs["langsmith_endpoint"]
    os.environ["LANGSMITH_API_KEY"] = configs["langsmith_api_key"]
    os.environ["LANGSMITH_PROJECT"] = configs["langsmith_project"]
    set_llm_cache(InMemoryCache())
    logging.getLogger("langchain.cache").setLevel(logging.DEBUG)


class DemoContentGenState(BaseModel):
    msgs: List[AnyMessage] = []
    results: Dict[str, str] = {}


class DemoBasicMathState(BaseModel):
    msgs: List[AnyMessage] = []               
    results: Dict[str, str] = {}   


async def demo_agent_content_gen(state: RouterState) -> Dict:
    print("Run content gen agent")
    name: str = state["name"]
    task: str = state["task"]
    global_state: State = state["state"]
    instruction: Instruction = [
        x for x in global_state.instructions if x.name == name
    ][0]
    agent_meta: BaseModel = global_state.agents_meta[task]
    agent_state_cls: Type = agent_meta.state
    out_msg: Dict = {
        "states": {
            task: {
                name: agent_state_cls()     
            }
        }
    }
    dest: agent_state_cls = out_msg["states"][task][name]
    agent_conf: Dict = agent_meta.configs 
    
    llm: BaseChatModel = global_state.llms[agent_conf["llm"]]
    msgs: List[Dict] = []
    sys_prompt: SystemMessage = SystemMessage(
        "You need generate the content following given instruction."
    )
    user_prompt: HumanMessage = HumanMessage(
        content=instruction.content
    )
    msgs.append(sys_prompt)
    msgs.append(user_prompt)
    resp: AIMessage = await llm.ainvoke(msgs)
    dest.msgs = msgs
    dest.results["final"] = resp.content
    return out_msg


async def demo_agent_basic_math(state: RouterState) -> Dict:
    print("Run basic math agent")
    name: str = state["name"]
    task: str = state["task"]
    global_state: State = state["state"]
    instruction: Instruction = [
        x for x in global_state.instructions if x.name == name
    ][0]
    agent_meta: BaseModel = global_state.agents_meta[task]
    agent_state_cls: Type = agent_meta.state
    out_msg: Dict = {
        "states": {
            task: {
                name: agent_state_cls()     
            }
        }
    }
    dest: agent_state_cls = out_msg["states"][task][name]
    agent_conf: Dict = agent_meta.configs 
    
    llm: BaseChatModel = global_state.llms[agent_conf["llm"]]
    msgs: List[Dict] = []
    sys_prompt: SystemMessage = SystemMessage(
        "You need generate the content following given instruction."
    )
    user_prompt: HumanMessage = HumanMessage(
        content=instruction.content
    )
    msgs.append(sys_prompt)
    msgs.append(user_prompt)
    resp: AIMessage = await llm.ainvoke(msgs)
    dest.msgs = msgs
    dest.results["final"] = resp.content
    return out_msg


async def main_demo() -> None:
    configs: Dict = json.loads(open(sys.argv[1], "r").read())
    print(configs)

    langchain_init(configs["langchain"])
    llms: Dict[str, BaseChatModel] = {
        x["model"]: init_chat_model(**x) for x in configs["llms"] 
    }
    instructions: List[Dict] = configs["demo"]["instructions"]
    agents_meta: List[AgentMeta] = [
        AgentMeta(
            name="content_gen",
            state=DemoContentGenState, 
            runner=demo_agent_content_gen,
            configs=configs["demo"]["agents"]["content_gen"]
        ),
        AgentMeta(
            name="basic_math",
            state=DemoBasicMathState,
            runner=demo_agent_basic_math,
            configs=configs["demo"]["agents"]["content_gen"]    
        )
    ]
    agents = agents_build(
        State, 
        agent_supervisor,
        edge_map,
        agents_meta
    )
    in_msg: Dict = {
        "llms": llms,
        "instructions": instructions,
        "agents_meta": {x.name: x for x in agents_meta},
        "states": None
    }
    out_msg: Dict = await agents.ainvoke(in_msg)
    print(out_msg["states"])
    return


if __name__ == "__main__":
    asyncio.run(main_demo())
