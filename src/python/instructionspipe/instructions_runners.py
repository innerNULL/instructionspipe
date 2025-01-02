# -*- coding: utf-8 -*-
# file: instructions_runner.py
# date: 2024-12-09


import pdb
import asyncio
import json
from typing import Union, Optional, List, Dict, Coroutine, Callable, Any, Set

from .instructions import instructions_to_output
from .instructions import instruction_is_empty
from .llm_cli import LlmCli
from .instructions import Instructions, Instruction


EMPTY_VAL: str = "N/A"


INVALID_VALS: Set[Optional[str]] = {
    EMPTY_VAL,
    None,
    "",
    " ",
    "NA",
    "N/A",
    "\n"
}


class InstructionsRunnerBase:
    def __init__(self):
        self.llm: Optional[LlmCli] = None
        self.instructions: Optional[Instructions] = None

    @classmethod
    def new_with_llm(
        cls,
        llm: LlmCli,
        instructions: Optional[Instructions]=None
    ):
        out = cls()
        out.llm = llm
        out.instructions = instructions
        return out
   
    def build_inputs(
        self, 
        input_data: Dict[str, str] | List | str, 
        instruction: Instruction
    ) -> Optional[str]:
        if isinstance(input_data, dict):
            if instruction.scope is not None:
                input_data = {
                    k: v for k, v in input_data.items() 
                    if k in instruction.scope and v not in INVALID_VALS
                }
            if len(input_data) == 0:
                return None
            return json.dumps(input_data, indent=2, ensure_ascii=False)
        elif isinstance(input_data, list):
            raise "Not supported condition 1"
            return json.dumps(input_data, indent=2, ensure_ascii=False)
        else:
            raise "Not supported condition 2"
            return input_data
 
    def build_sys_msg(
        self, 
        in_data: Dict[str, str], 
        instruction: Instruction
    ) -> str:
        out: str = ""
        if instruction.content is not None:
            out += "## Instruction\n%s\n\n" % instruction.content
        if instruction.role is not None:
            out += "## Your Role\n%s\n\n" % instruction.role
        if instruction.input_desc is not None:
            out += "## Your Given Input\n%s\n\n" % instruction.input_desc
        if instruction.output_desc is not None:
            out += "## The Extected Output\n%s\n\n" % instruction.output_desc
        return out

    def build_user_msg(
        self, 
        in_data: Dict[str, str], 
        instruction: Instruction
    ) -> str:
        usr_input: Optional[str] = self.build_inputs(in_data, instruction)
        if usr_input is not None:
            return "%s" % usr_input
        else:
            return None

    def init_chatml(
        self, 
        in_data: Dict[str, str], 
        instruction: Instruction
    ) -> List[ Optional[Dict[str, str]] ]:
        out: List[Dict[str, str]] = [
            {"role": "system", "content": None}, 
            {"role": "system", "content": None}
        ]
        if not instruction_is_empty(instruction):
            out = [
                {
                    "role": "system",
                    "content": self.build_sys_msg(in_data, instruction)
                },
                {
                    "role": "user",
                    "content": self.build_user_msg(in_data, instruction)
                }
            ]
        if "mistral" in self.llm.model.lower():
            out[0]["role"] = "user"
            out = [
                out[0],
                {"role": "assistant", "content": "Ok."},
                out[1]
            ]
        return out

    def init_instructions_chatml(
        self, 
        in_data: Dict[str, str],
        instructions: Optional[Instructions]=None, 
        fn: Optional[Callable]=None
    ) -> Instructions:
        if fn is None:
            fn = self.init_chatml
        if instructions is None:
            instructions = self.instructions
        for i, instruction in enumerate(instructions.instructions):
            instructions.instructions[i].msgs = fn(in_data, instruction)
        return instructions
    
    async def async_run(
        self,
        prev_instructions: Optional[Instructions]=None,
        instructions: Optional[Instructions]=None
    ) -> Instructions:
        instructions = self.init_instructions_chatml(
            prev_instructions.result, instructions
        )
        chatmls: List[ List[ Optional[Dict[str, str]] ] ] = [
            x.msgs for x in instructions.instructions
        ]
        tasks: List[Coroutine] = [
            self.llm.async_run(chatml[-1], chatml[:-1]) 
            for chatml in chatmls
        ]
        resps: List[Optional[ChatCompletion]] = await asyncio.gather(*tasks)
        for i in range(len(instructions.instructions)):
            instructions.instructions[i].msgs.append({
                "role": "assistant", 
                "content": (
                    resps[i].choices[0].message.content 
                    if resps[i] is not None else EMPTY_VAL
                )
            })
            instructions.instructions[i].finished = True
        instructions_to_output(instructions)
        return instructions
