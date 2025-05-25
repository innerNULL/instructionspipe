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


class Instruction(BaseModel):
    # Instruction name
    name: str
    # Input description
    input_desc: Optional[str] = None
    # Output description
    output_desc: Optional[str] = None
    # Output format
    output_fmt: Optional[str] = None
    # Misc contents
    content: Optional[str] = None
    # Role
    role: Optional[str] = None
    # In-context examples
    #examples: Optional[List[InContextExample]] = None
    # External knowledges
    knowledge: Optional[List[str]] = None
    # Input fields needed by this instruction
    scope: Optional[List[str]] = None
    # ChatML messages
    msgs: Optional[List[Dict[str, str]]] = None
    # Status, finished or not
    finished: bool = False
    # Stage number of this instruction in full pipeline
    stage: Optional[int] = None
    # Pipeline's session ID
    session_id: Optional[str] = None
    # Instruction ID
    instruction_id: Optional[str] = None
    # LLM
    model: Optional[str] = None
    # Task category
    task: Optional[str] = None


class AgentMeta(BaseModel):
    name: str
    state: Type
    runner: Callable
    configs: Optional[Dict] = None         

def merge_dicts(
    a: Dict[str, Dict[str, BaseModel]], 
    b: Dict[str, Dict[str, BaseModel]]
) -> Dict[str, Dict[str, BaseModel]]:
    if a is None:
        return b
    result = a.copy()
    for key, value in b.items():
        if key in result:
            result[key].update(value)
        else:
            result[key] = value
    return result


class State(BaseModel):
    llms: Optional[Dict[str, BaseChatModel]] = None
    instructions: Optional[List[Instruction]] = None
    #states: Optional[Dict[str, Dict[str, BaseModel]]] = None
    states: Annotated[Optional[Dict[str, Dict[str, BaseModel]]], merge_dicts] = None
    agents_meta: Optional[Dict[str, AgentMeta]] = None


class RouterState(BaseModel):
    name: str 
    task: str
    state: State


def langchain_init(configs: Dict) -> None:
    os.environ["LANGSMITH_TRACING"] = str(configs["langsmith_tracing"]).lower()
    os.environ["LANGSMITH_ENDPOINT"] = configs["langsmith_endpoint"]
    os.environ["LANGSMITH_API_KEY"] = configs["langsmith_api_key"]
    os.environ["LANGSMITH_PROJECT"] = configs["langsmith_project"]
    set_llm_cache(InMemoryCache())
    logging.getLogger("langchain.cache").setLevel(logging.DEBUG)


def agent_supervisor(state: State) -> Dict:
    return {}


def edge_map(state: State) -> List[Send]:
    instructions: List[Instruction] = state.instructions
    routers: List[Send] = []
    for instruction in instructions:
        name: str = (
            instruction.name if instruction.name is not None 
            else "default"
        )
        task: str = instruction.task
        agent_meta: AgentMeta = state.agents_meta[task]
        msg: Dict = {
            "task": task,
            "name": name,
            "state": state
        }
        router: Send = Send(task, msg)
        routers.append(router)
    return routers


def agents_build(
    state: Type,
    agent_supervisor: Callable,
    edge_map: Callable,
    agents_meta: List[AgentMeta],
    supervisor_name: str="supervisor"
) -> CompiledStateGraph:
    builder: StateGraph = StateGraph(state)
    builder.add_node(supervisor_name, agent_supervisor)
    builder.add_edge(START, supervisor_name)
    for agent_meta in agents_meta:
        name: str = agent_meta.name
        runner: Callable = agent_meta.runner
        state: Type = agent_meta.state
        builder.add_node(name, runner)
        builder.add_edge(name, END)
    builder.add_conditional_edges(supervisor_name, edge_map)
    graph: CompiledStateGraph = builder.compile()
    return graph


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
