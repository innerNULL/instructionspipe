# -*- coding: utf-8 -*-
# file: tableqa.py
# date: 2025-05-16

import os
import sys
import pdb
import json
import asyncio
import io
import tempfile
import subprocess
import uuid
import logging
from pydantic import BaseModel
from typing import Dict, List, Optional, Annotated, Literal, Callable, Union, Type, Any
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
from langgraph.types import Command
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model
from tqdm import tqdm

logging.basicConfig(
    #level=logging.DEBUG,  # Set the minimum logging level
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),  # Log messages to a file
        logging.StreamHandler()         # Log messages to the console
    ]
)


PROMPT_SYS: str = \
"""
You will be given a instruction and a free text.
You need to perform the instruction on the given free text to get the 
answer by writting Python code and run it.

You should output either:
- a Python code snippet that provides the solution to the task, or a step 
  towards the solution. Any output you want to extract from the code should 
  be printed to the console with `print` (need alse print some description
  of the output, like what the output is). Code should be output in a fenced 
  code block. The code snippet must be always enclosed with <code> and </code>.

- text to be shown directly to the user, if you want to ask for more 
  information or provide the final answer.

In addition to the Python Standard Library, you can use the following 
Python modules:
* pandas
"""


LOGGER = logging.getLogger(__name__)


class TableQaCodeActState(BaseModel):
    inputs: Optional[str | Dict] = None
    instruction: Optional[str] = None
    prompt_sys: str = PROMPT_SYS 
    msgs: List[AnyMessage] = []
    max_rounds: int = 5


class State(BaseModel):
    llms: Dict[str, BaseChatModel] = {}
    tableqa_codeact: Optional[TableQaCodeActState] = None 


def langchain_init(configs: Dict) -> None:
    os.environ["LANGSMITH_TRACING"] = str(configs["langsmith_tracing"]).lower()
    os.environ["LANGSMITH_ENDPOINT"] = configs["langsmith_endpoint"]
    os.environ["LANGSMITH_API_KEY"] = configs["langsmith_api_key"]
    os.environ["LANGSMITH_PROJECT"] = configs["langsmith_project"]


def tag_extract(text: str, tag_start: str, tag_end: str) -> Optional[str]:
    out: Optional[str] = None
    if tag_start not in text or tag_end not in text:
        return out
    start_idx: int = text.index(tag_start) + len(tag_start)
    end_idx: int = text.index(tag_end)
    if start_idx >= end_idx:
        return out
    return text[start_idx:end_idx]


def exec_err(outputs: str) -> bool:
    if outputs[:9] == "Traceback":
        return True
    return False


def sandbox_run(code: str, workspace_dir: Optional[str] = None) -> str:
    """
    This function creates a Python script under `workspace_dir` with the given `code`.
    If `workspace_dir` is None, it creates a temporary directory with a random hash.
    Then it runs the created Python script using the current Python runtime.
    Returns the stdout of the script execution, including any errors or tracebacks.
    """
    # Define variable types
    workspace_dir: str = workspace_dir or tempfile.mkdtemp(prefix="sandbox_")
    script_path: str = os.path.join(workspace_dir, f"script_{uuid.uuid4().hex}.py")

    try:
        # Write the code to the script file
        with open(script_path, 'w') as script_file:
            script_file.write(code)
        
        # Run the script using the current Python runtime
        process = subprocess.Popen(
            [os.sys.executable, script_path],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT  # Redirect stderr to stdout
        )
        stdout, _ = process.communicate()  # Capture the combined output
        return stdout.strip("\n").strip(" ").strip("\n").strip(" ")
    
    finally:
        # Clean up the script file
        if os.path.exists(script_path):
            os.remove(script_path)


async def agent_codeact(state: State) -> Command:
    curr_state: TableQaCodeActState = state.tableqa_codeact
    llm: BaseChatModel = state.llms["gpt-4o-mini"]
    msgs: List[AnyMessage] = curr_state.msgs
    sys_prompt: str = curr_state.prompt_sys
    inputs: Dict = curr_state.inputs
    instruction: str = curr_state.instruction
    max_rounds: int = curr_state.max_rounds
    if len(msgs) == 0:
        msgs.append(SystemMessage(content=sys_prompt))
        msgs.append(HumanMessage(
            content="<free_text>\n{}\n</free_text>\n<instruction>\n{}\n</instruction>".format(
                json.dumps(inputs, indent=2) if isinstance(inputs, dict) else inputs,
                instruction
            )
        ))
    resp: AIMessage = await llm.ainvoke(msgs)
    msgs.append(resp)
    code: Optional[str] = tag_extract(resp.content, "<code>", "</code>")
    if not code or max_rounds == 0:
        return Command(goto=END, update={}) 
    else:
        run_results: str = sandbox_run(code)
        failed: bool = exec_err(run_results)
        if failed:
            msgs.append(HumanMessage(
                content="Something wrong when running the code: \n%s" % run_results
            ))
            return Command(
                goto="agent_codeact", 
                update={
                    "tableqa_codeact": {
                        "msgs": msgs,
                        "max_rounds": max_rounds - 1
                    }
                }
            )
        else:
            msgs.append(HumanMessage(content=run_results))
            LOGGER.info("CodeAct result: {}".format(run_results))
            return Command(
                goto=END, 
                update={
                    "tableqa_codeact": {
                       "msgs": msgs,
                       "max_rounds": max_rounds - 1   
                    }
                }
            )    


def agents_build(
    state: Type=State, 
    agent_codeact: Callable=agent_codeact,
) -> CompiledStateGraph:
    builder: StateGraph = StateGraph(state)
    builder.add_node("agent_codeact", agent_codeact)
    builder.add_edge(START, "agent_codeact")
    graph: CompiledStateGraph = builder.compile()
    return graph


async def main_inf_offline() -> None:
    configs: Dict = json.loads(open(sys.argv[2], "r").read())
    print(configs)
    llm_configs: Dict = configs["llms"] 
    in_data_path: str = configs["in_data_path"]
    inputs_col: str = configs["inputs_col"]
    instruction_col: str = configs["query_task_col"]

    langchain_init(configs["langchain"])
    llms: Dict[str, BaseChatModel] = {
        x["model"]: init_chat_model(**x) for x in llm_configs
    }
    agents = agents_build()
    samples: List[Dict] = [
        json.loads(x) for x in open(in_data_path, "r").read().split("\n")
        if x not in {""}
    ]
    for sample in tqdm(samples):
        inputs: Dict = {
            "llms": llms, 
            "tableqa_codeact": {
                "inputs": sample[inputs_col], 
                "instruction": sample[instruction_col]
            }
        }
        results: Dict = await agents.ainvoke(inputs)
    return


async def serving_http() -> None:
    return


if __name__ == "__main__":
    scenario: str = sys.argv[1]
    if scenario == "inf_offline":
        asyncio.run(main_inf_offline())
    elif scenario == "serving_http":
        asyncio.run(main_serving_http())              
    else:
        raise "Woring arguments"
