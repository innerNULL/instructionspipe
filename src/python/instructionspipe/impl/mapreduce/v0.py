# -*- coding: utf-8 -*-
# file: v0.py
# date: 2024-12-16


import pdb
import asyncio
import traceback
import copy
import json
import uuid
from tqdm import tqdm
from typing import Union, Optional, List, Dict, Coroutine, Callable, Any
from pydantic import BaseModel
from openai import AsyncOpenAI, AsyncAzureOpenAI
from openai import ChatCompletion

from ...utils import json2str_kv
from ...instructions import instructions_init_by_configs
from ...instructions import instructions_to_output
from ...instructions import instructions_to_md
from ...instructions import multi_instructions_collect
from ...instructions import Instruction, Instructions
from ...instructions_runners import InstructionsRunnerBase
from ...llm_cli import LlmCli


async def run_with_configs(
    llm: LlmCli,
    inputs: Dict, 
    map_conf: List[Dict], 
    reduce_conf: List[Dict]
) -> Dict:
    mapper: InstructionsRunnerBase = InstructionsRunnerBase.new_with_llm(
        llm=llm, 
        instructions=instructions_init_by_configs(map_conf)
    )
    reducer: InstructionsRunnerBase = InstructionsRunnerBase.new_with_llm(
        llm=llm, 
        instructions=instructions_init_by_configs(reduce_conf)
    )
    init_instructions: Instructions = Instructions(
        instructions=[], 
        result=json2str_kv(inputs), 
        finished=True
    )
    map_instructions: Instructions = (
        await mapper.async_run(init_instructions)
    )
    reduce_instructions: Instructions = (
        await reducer.async_run(map_instructions)
    )
    
    session_id: str = str(uuid.uuid4())
    for map_instruction in map_instructions.instructions:
        map_instruction.session_id = session_id
        map_instruction.stage = 1
    for reduce_instruction in reduce_instructions.instructions:
        reduce_instruction.session_id = session_id
        reduce_instruction.stage = 2

    outputs: Dict = {
        "map_results": map_instructions.result, 
        "reduce_results": reduce_instructions.result,
        "result": instructions_to_md(reduce_instructions),
        "chatmls": multi_instructions_collect(
            [map_instructions, reduce_instructions]
        )
    }
    return outputs
