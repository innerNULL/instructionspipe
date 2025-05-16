# -*- coding: utf-8 -*-
# file: instructions_runner.py
# date: 2024-12-09


import pdb
import asyncio
import json
from typing import Union, Optional, List, Dict, Coroutine, Callable, Any, Set

from .instructions import instructions_to_output
from .instructions import instruction_is_empty
from .instructions import instruction_to_sys_prompt
from .llm_cli import LlmCli
from .instructions import Instructions, Instruction
from .constants import EMPTY_VAL
from .constants import INVALID_VALS
from .logger import get_logger


LOGGER = get_logger(__name__)


class InstructionsRunnerBase:
    def __init__(self):
        self.llms: Optional[Dict[str, LlmCli]] = None
        self.instructions: Optional[Instructions] = None
        self.model_default: Optional[str] = None

    @classmethod
    def new_with_llm(
        cls,
        llms: Dict[str, LlmCli],
        instructions: Optional[Instructions]=None
    ):
        out = cls()
        out.llms = llms
        out.instructions = instructions
        out.model_default = [x for x in llms.keys()][0]
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
    ) -> List[ Dict[str, Optional[str]] ]:
        out: List[Dict[str, str]] = [
            {"role": "system", "content": None}, 
            {"role": "system", "content": None}
        ]
        model: str = (
            instruction.model if instruction.model is not None 
            else self.model_default
        )
        if not instruction_is_empty(instruction):
            out = [
                {
                    "role": "system",
                    "content": instruction_to_sys_prompt(instruction)
                },
                {
                    "role": "user",
                    "content": self.build_user_msg(in_data, instruction)
                }
            ]
        else:
            LOGGER.warning("The instruction is empty, so do ChatML")
        if (
            "mistral" in model.lower()
            or "gemma" in model.lower()
        ):
            LOGGER.info("Removing system prompt for specific LLMs")
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
        chatmls: List[ List[ Dict[str, Optional[str]] ] ] = [
            x.msgs for x in instructions.instructions
        ]
        llms: List[LlmCli] = [
            self.llms[x.model] if x.model is not None
            else self.llms[self.model_default] 
            for x in instructions.instructions
        ]
        assert(len(chatmls) == len(llms))
        tasks: List[Coroutine] = [
            llms[i].async_run(chatmls[i][-1], chatmls[i][:-1]) 
            for i in range(len(chatmls))
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
