# -*- coding: utf-8 -*-
# file: instructions.py.py
# date: 2024-12-09


from typing import Union, Optional, List, Dict, Coroutine, Callable, Any, Set
from pydantic import BaseModel

from .constants import INVALID_VALS


class Instruction(BaseModel):
    name: str
    input_desc: Optional[str] = None
    output_desc: Optional[str] = None
    content: Optional[str] = None
    role: Optional[str] = None
    scope: Optional[List[str]] = None
    msgs: Optional[List[Dict[str, str]]] = None
    finished: bool = False
    stage: Optional[int] = None
    session_id: Optional[str] = None


class Instructions(BaseModel):
    instructions: List[Instruction]
    result: Optional[Dict[str, str]] = None
    finished: bool = False


def instruction_to_sys_prompt(instruction: Instruction) -> str:
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

