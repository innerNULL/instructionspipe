# -*- coding: utf-8 -*-
# file: self_verification_mr.py
# date: 2024-11-09


import pdb
import asyncio
import sys
import os
import copy
import json
from typing import Union, Optional, List, Dict, Coroutine, Callable, Any
from pydantic import BaseModel
from openai import OpenAI, AsyncOpenAI
from openai import ChatCompletion


INIT_GEN_SCHEMA: Dict = \
{
    "type": "json_schema",
    "json_schema": {
        "name": "instruction_generated_elements_schema",
        "schema": {
            "type": "object",
            "properties": {
                "content": {"type": "string"}
            }, 
            "required": ["content"],
            "additionalProperties": False
        },
        "strict": True
    }
}


def llm_resp_json_clean(in_json: str) -> str:
    return in_json.replace("```json", "").replace("```", "")


def any_to_str(in_data: Any) -> str:
    if isinstance(in_data, str):
        return in_data
    elif isinstance(in_data, int) or isinstance(in_data, float):
        return str(in_data)
    elif isinstance(in_data, list) or isinstance(in_data, dict):
        return json.dumps(
            in_data, ensure_ascii=False, indent=2
        )
    else:
        raise Exception("Failed convert to string")


class LlmCli:
    def __init__(self):
        self.cli: Optional[OpenAI] = None
        self.async_cli: Optional[AsyncOpenAI] = None
        self.model: Optional[str] = None
        self.seed: Optional[int] = None
        self.temperature: Optional[float] = None
        self.top_p: Optional[float] = None

    @classmethod
    def new(
        cls, 
        model: str, 
        api_key: str,
        api_url: str, 
        seed: int=2, 
        temperature: float=0.0, 
        top_p: float=0.01
    ):
        out = cls()
        out.cli =  OpenAI(api_key=api_key, base_url=api_url)
        out.async_cli = AsyncOpenAI(api_key=api_key, base_url=api_url)
        out.model = model
        out.seed = seed
        out.temperature = temperature
        out.top_p = top_p
        return out

    @classmethod
    def new_with_configs(cls, configs: Dict):
        return cls.new(
            model=configs["model"], 
            api_key=configs["api_key"],
            api_url=configs["api_url"],
            temperature=configs["temperature"],
            seed=configs["seed"]
        )

    def run(self, 
        msg: Union[str, Dict], 
        prefix: Union[Dict, List[Dict]]=None, 
        json_schema: Optional[Dict]=None
    ) -> ChatCompletion:
        if isinstance(msg, str):
            msg = {"role": "user", "content": msg}
        if prefix is None:
            prefix = []
        if isinstance(prefix, dict):
            prefix = [prefix]
        return self.cli.chat.completions.create(
            model=self.model, 
            messages=prefix + [msg],
            seed=self.seed,
            temperature=self.temperature,
            top_p=self.top_p,
            response_format=json_schema
        )

    async def async_run(
        self, 
        msg: Union[str, Dict],
        prefix: Union[Dict, List[Dict]]=None,
        json_schema: Optional[Dict]=None
    ):
        if isinstance(msg, str):
            msg = {"role": "user", "content": msg}
        if prefix is None:
            prefix = []
        if isinstance(prefix, dict):
            prefix = [prefix]
        return await self.async_cli.chat.completions.create(
            model=self.model, 
            messages=prefix + [msg],
            seed=self.seed,
            temperature=self.temperature,
            top_p=self.top_p,
            response_format=json_schema
        )
   

class Instruction(BaseModel):
    name: str
    content: Optional[str]=None
    scope: Optional[List[str]]=None
    msgs: Optional[List[Dict[str, str]]]=None


class InstructionsMapper:
    def __init__(self):
        self.llm: Optional[LlmCli] = None
        self.instructions: Optional[List[Instruction]] = None
        self.role: Optional[str] = None

    @classmethod
    def new_with_llm(
        cls,
        llm: LlmCli,
        role: str,
        instructions: Optional[List[Instruction]]=None
    ):
        out = cls()
        out.llm = llm
        out.role = role
        out.instructions = instructions
        return out


class SelfVerifiedMapper(InstructionsMapper):
    def build_extraction_chatml(
        self,
        input_text: str,
        instruction: Instruction
    ) -> Instruction:
        msgs: List[Dict] = [
            {
                "role": "system",
                "content": (
                    "__ROLE__\n"
                    "Your task is to extract elements from given text following the given instruction."
                    "The instruction you need to follow is: __INSTRUCTION__\n"
                    "Your output must be a JSON array of string, which contains the "
                    "elements you extracted from given text by following given instruction. \n"
                    "You have to only return the JSON array of string without anything else."
                )\
                    .replace("__ROLE__", self.role)\
                    .replace("__INSTRUCTION__", instruction.content)
            },
            {
                "role": "user",
                "content": input_text
            }
        ]
        instruction.msgs = msgs
        return instruction

    def build_omission_chatml(
        self,
        input_text: Optional[str],
        instruction: Instruction
    ) -> Instruction:
        prompt: Dict = {
            "role": "user",
            "content": (
                "Based on given text, "
                "check which information are missed in above result, "
                "do complementation if any missing information found.\n"
                "\n"
                "You should only return JSON array of string as above, "
                "with the complementation has been done."
            )
        }
        instruction.msgs.append(prompt)
        return instruction

    def build_evidence_chatml(
        self, 
        input_text: Optional[str],
        instruction: Instruction
    ) -> Instruction:
        prompt: Dict = {
            "role": "user",
            "content": (
                "Find the span of text which corresponds to each instruction response listed above. "
                "If no evidence is found, write \"No evidence can support this statement.\".\n"
                "\n"
                "Your output should be a JSON array of object, "
                "each object is a JSON contains 'content' and 'evidence', "
                "the value of 'content' must be one of the elements listed above, "
                "and the value of 'evidence' is the evidence you extracted from given text, "
                "which can support the value of 'content', "
                "which have to be a text span sourced from given text."
            )
        }
        instruction.msgs.append(prompt)
        return instruction

    def build_chatmls(
        self, 
        input_text: str,
        instructions: List[Instruction], 
        fn: Callable
    ) -> List[Instruction]:
        for instruction in instructions:
            instruction = fn(input_text, instruction)
        return instructions
    
    async def async_run_extraction(
        self,
        input_text: str,
        instructions: List[Instruction]
    ) -> List[Instruction]:
        instructions = self.build_chatmls(
            input_text, instructions, self.build_extraction_chatml
        )
        tasks: List[Coroutine] = [
            self.llm.async_run(chatml[-1], chatml[:-1]) for chatml in 
            [x.msgs for x in instructions]
        ]
        resps: List[ChatCompletion] = await asyncio.gather(*tasks)
        for i in range(len(instructions)):
            instructions[i].msgs.append({
                "role": "assistant", 
                "content": resps[i].choices[0].message.content
            })
        return instructions

    async def async_run_omission(
        self, 
        instructions: List[Instruction]
    ) -> List[Instruction]:
        instructions = self.build_chatmls(
            None, instructions, self.build_omission_chatml
        )
        tasks: List[Coroutine] = [
            self.llm.async_run(chatml[-1], chatml[:-1]) for chatml in
            [x.msgs for x in instructions]
        ]
        resps: List[ChatCompletion] = await asyncio.gather(*tasks)
        for i in range(len(instructions)):
            instructions[i].msgs.append({
                "role": "assistant", 
                "content": resps[i].choices[0].message.content
            })
        return instructions

    async def async_run_evidence(
        self,
        instructions: List[Instruction]
    ) -> List[Instruction]:
        instructions = self.build_chatmls(
            None, instructions, self.build_evidence_chatml
        )
        tasks: List[Coroutine] = [
            self.llm.async_run(chatml[-1], chatml[:-1]) for chatml in
            [x.msgs for x in instructions]
        ]
        resps: List[ChatCompletion] = await asyncio.gather(*tasks)
        for i in range(len(instructions)):
            instructions[i].msgs.append({
                "role": "assistant", 
                "content": resps[i].choices[0].message.content
            })
        return instructions

    async def async_run_prune_rule_based(
        self,
        instructions: List[Instruction]
    ) -> List[Instruction]:
        return instructions

    async def async_run(
        self,
        input_text: str,
        instructions: Optional[List[Instruction]]=None
    ) -> List[Instruction]:
        if instructions is None:
            instructions = copy.deepcopy(self.instructions)
        assert(instructions is not None)
        assert(len(instructions) > 0)
        instructions = (
            await self.async_run_extraction(input_text, instructions)
        )
        instructions = await self.async_run_omission(instructions)
        instructions = await self.async_run_evidence(instructions)
        instructions = await self.async_run_prune_rule_based(instructions)
        return instructions


class InstructionsReducer:
    def __init__(self):
        self.llm: Optional[LlmCli] = None
        self.role: Optional[str] = None
        self.groups: Optional[List[Instruction]]=None

    @classmethod
    def new_with_llm(
        cls,
        llm: LlmCli,
        role: str,
        groups: Optional[List[Instruction]]=None
    ):
        out = cls()
        out.llm = llm
        out.role = role
        out.groups = groups
        return out


class RewritingReducer(InstructionsReducer):
    def build_chatml(
        self, 
        instructions: List[Instruction],
        group: Instruction
    ) -> Instruction:
        group.msgs = []
        target_instructions: List[Instruction] = [
            x for x in instructions if x.name in group.scope
        ]
        target_data: str = ""
        for instruction in target_instructions:
            name: str = instruction.name
            final_resp: str = llm_resp_json_clean(
                instruction.msgs[-1]["content"]
            )
            map_output: List[str] = [
                x["content"] for x in json.loads(final_resp)
            ]
            target_data += (
                "<__NAME__>\n"
                "__CONTENT__\n"
                "</__NAME__>\n\n"
            )\
                .replace("__NAME__", name)\
                .replace("__CONTENT__", json.dumps(map_output, indent=2))
   
        group.msgs = [
            {
                "role": "system",
                "content": (
                    "__ROLE__ \n"
                    "Your task is to rewrite given semi-structured data "
                    "into natural language description. "
                )\
                    .replace("__ROLE__", self.role)\
            },
            {
                "role": "user",
                "content": target_data
            }
        ]
        return group

    async def async_run_group(
        self, 
        instructions: List[Instruction],
        group: Instruction
    ) -> Instruction:
        group = self.build_chatml(instructions, group)
        resp: ChatCompletion = await self.llm.async_run(
            group.msgs[-1], group.msgs[:-1]
        ) 
        group.msgs.append(
            {
                "role": "assistant",
                "content": resp.choices[0].message.content
            }
        )
        return group

    async def async_run(
        self,
        instructions: List[Instruction],
        groups: Optional[List[Instruction]]=None
    ) -> List[Instruction]:
        if groups is None:
            groups = copy.deepcopy(self.groups)
        assert(groups is not None)
        assert(len(groups) > 0)
        tasks: List[Coroutine] = [
            self.async_run_group(instructions, x) for x in groups
        ]
        return await asyncio.gather(*tasks)


class SelfVerifiedMR:
    def __init__(self):
        self.llm: Optional[LlmCli] = None
        self.map_role: Optional[str] = None
        self.instructions: Optional[List[Instruction]] = None
        self.reduce_role: Optional[str] = None
        self.groups: Optional[List[Instruction]] = None
        self.mapper: Optional[SelfVerifiedMappe] = None
        self.reducer: Optional[RewritingReducer] = None

    @classmethod
    def new_with_llm_and_mr_configs(
        cls,
        llm: LlmCli,
        map_conf: Dict, 
        reduce_conf: Dict
    ):
        out = cls()
        out.llm = llm
        out.mapper = SelfVerifiedMapper.new_with_llm(
            llm=llm,
            role=map_conf["role"],
            instructions=[
                Instruction.parse_obj(x) for x in map_conf["instructions"]
            ]
        )
        out.reducer = RewritingReducer.new_with_llm(
            llm=llm,
            role=reduce_conf["role"],
            groups=[
                Instruction.parse_obj(x) for x in reduce_conf["instructions"]
            ]
        )
        return out

    async def async_run(self, input_text: str) -> Dict:
        out: Dict = {}
        instructions: List[Instruction] = (
            await self.mapper.async_run(input_text)
        )
        groups: List[Instruction] = (
            await self.reducer.async_run(instructions)
        )
        result: str = ""
        for group in groups:
            name: str = group.name
            final_resp: str = group.msgs[-1]["content"]
            result += "# {}\n".format(name)
            result += "{}\n".format(final_resp)
            result += "\n"
        out["result"] = result
        return out


async def main() -> None:
    configs: Dict = json.loads(open(sys.argv[1], "r").read())
    print(configs)
    in_data_path: str = configs["in_data_path"]
    in_text_cols: str = configs["in_text_cols"]
    output_col: str = configs["output_col"]
    map_conf: Dict = configs["runner"]["map"]
    reduce_conf: Dict = configs["runner"]["reduce"]
 
    llm: LlmCli = LlmCli.new_with_configs(configs["llm"])
    runner: SelfVerifiedMR = SelfVerifiedMR.new_with_llm_and_mr_configs(
        llm, map_conf, reduce_conf
    ) 

    # Check
    print("Testing LLM's response")
    test_resp: Coroutine = llm.async_run("Hi")
    print("Running 'Hi'")
    test_result: str = (await test_resp).choices[0].message.content
    print(test_result)
    print("Testing finished")

    in_samples: List[Dict] = [
        json.loads(x) for x in open(in_data_path, "r").read().split("\n")
        if x not in {""}
    ]
    for in_sample in in_samples:
        input_text: str = ""
        for col in in_text_cols:
            input_text += "# %s\n" % col
            input_text += any_to_str(in_sample[col])
            input_text += "\n\n"
        output: Dict = await runner.async_run(input_text)
        print(output["result"])
    return


if __name__ == "__main__":
    asyncio.run(main())
