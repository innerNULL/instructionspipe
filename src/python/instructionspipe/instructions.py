# -*- coding: utf-8 -*-
# file: instructions.py.py
# date: 2024-12-09


import json
import pdb
from typing import Union, Optional, List, Dict, Coroutine, Callable, Any, Set
from pydantic import BaseModel
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStore

from .constants import INVALID_VALS


class InContextExample(BaseModel):
    in_text: str
    out_text: str


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
    examples: Optional[List[InContextExample]] = None
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


class Instructions(BaseModel):
    instructions: List[Instruction]
    result: Optional[Dict[str, str]] = None
    finished: bool = False


def instruction_postproc_ic_ralm(
    instruction: Optional[Instruction], 
    input_data: str | Dict[str, str],
    retriever: Optional[BaseRetriever | Dict[str, BaseRetriever]],
    keys: Optional[List[str]]=None,
    k: int=1,
    embedding: Optional[Embeddings]=None,
    search_args: Optional[Dict]={},
    example_input_col: str="in_text",
    example_output_col: str="out_text"
) -> Optional[Instruction]:
    """
    Ref 
    * https://arxiv.org/abs/2302.00083

    Args:
        instruction: Target `Instruction` instance.
        input_data: Input data of this instruction
        retriever: LangChain's `BaseRetriever`.
        k: Top-k retrieval parameter.
    """
    if instruction is None:
        return instruction
    if keys is None and isinstance(input_data, dict):
        keys = list(input_data.keys())
    
    query: str = ""
    if isinstance(input_data, dict):
        query = json.dumps(
            {k: v for k, v in input_data.items() if k in keys},
            indent=2
        )
    elif isinstance(input_data, str):
        query = input_data
    else:
        raise Exception("Type error caused by `input_data`")

    instruction.examples = []
    if isinstance(retriever, dict):
        if instruction.name in retriever:
            retriever = retriever[instruction.name]
        elif (
            instruction.instruction_id is not None
            and instruction.instruction_id in retriever
        ):
            retriever = retriever[instruction.instruction_id]
        else:
            raise Exception("Invalid `retriever`")
    results: List[Document] = retriever.invoke(query, k=k)
    icl_examples_desc: str = ""
    for i, doc in enumerate(results[:k]):
        data: Dict = results[i].metadata
        example: InContextExample = InContextExample(
            in_text=data[example_input_col],
            out_text=data[example_output_col]
        )
        instruction.examples.append(example)
    return instruction


def instruction_to_sys_prompt_v0(instruction: Instruction) -> str:
    #out: str = "Following are the details of the task you need to finish.\n"
    out: str = ""
    if instruction.content is not None:
        out += "## Instruction\n%s\n\n" % instruction.content
    if instruction.role is not None:
        out += "## Your Role\n%s\n\n" % instruction.role
    if instruction.input_desc is not None:
        out += "## Your Given Input\n%s\n\n" % instruction.input_desc
    if instruction.output_desc is not None:
        out += "## The Expected Output\n%s\n\n" % instruction.output_desc
    return out


def instruction_to_sys_prompt(instruction: Instruction) -> str:
    out: str = "Following are the details of the task you need to finish.\n\n"
    if instruction.role is not None:
        out += "## Your Role\n%s\n\n" % instruction.role
    if instruction.input_desc is not None:
        out += "## Your Given Input\n%s\n\n" % instruction.input_desc
    if instruction.output_desc is not None:
        out += "## The Expected Output\n%s\n\n" % instruction.output_desc
    if instruction.output_fmt is not None:
        out += "## Output Format\n%s\n\n" % instruction.output_fmt
    if instruction.content is not None:
        out += "## Instruction Misc\n%s\n\n" % instruction.content
    if instruction.knowledge is not None and len(instruction.knowledge) > 0:
        out += "## External Knowledge\n"
        out += "\n".join(instruction.knowledge)
        out += "\n\n"
    if instruction.examples is not None and len(instruction.examples) > 0:
        out += "## Examples\n"
        for i, example in enumerate(instruction.examples):
            out += "Example %i:\n" % i
            out += "Input: %s\n" % example.in_text
            out += "Output: %s\n" % example.out_text
            out += "\n"
        out += "\n"
    return out


def instructions_init_by_configs(
    configs: List[Dict]
) -> Instructions:
    return Instructions(
        instructions=[
            Instruction.parse_obj(x) for x in configs
        ],
        result=None,
        finished=False
    )


def instructions_to_output(
    instructions: Instructions
) -> Instructions:
    for instruction in instructions.instructions:
        if instruction.finished == False:
            instructions.result = None
            break
        name: str = instruction.name
        val: str = instruction.msgs[-1]["content"]
        """
        val: Dict | List | str = instruction.msgs[-1]["content"]
        try:
            val = json.loads(llm_resp_json_clean(val))
        except Exception as e:
            pass
        """
        if instructions.result is None:
            instructions.result = {}
        instructions.result[name] = val
    instructions.finished = True
    return instructions


def instructions_to_md(instructions: Instructions) -> str:
    result: str = ""
    for instruction in instructions.instructions:
        name: str = instruction.name
        final_resp: str = instruction.msgs[-1]["content"]
        result += "# {}\n".format(name)
        result += "{}\n".format(final_resp)
        result += "\n"
    return result
  

def instruction_is_empty(instruction: Instruction) -> bool:
    if (
        instruction.input_desc in INVALID_VALS 
        and (
            instruction.output_desc in INVALID_VALS 
            and instruction.content in INVALID_VALS
        )
    ):
        return True
    return False


def instructions_collect(instructions: Instructions) -> List[Dict]:
    out: List[Dict] = []
    for instruction in instructions.instructions:
        out.append(instruction.dict())
    return out


def multi_instructions_collect(multi_instructions: List[Instructions]) -> List[Dict]:
    out: List[Dict] = []
    for instructions in multi_instructions:
        out += instructions_collect(instructions)
    return out

