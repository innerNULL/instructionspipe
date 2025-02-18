# -*- coding: utf-8 -*-
# file: mr_pipeline.py
# date: 2024-11-09
#
# Run example:
# python ./bin/mr_pipeline.py demo_configs/pipeline_mr/ehr.json
# python ./bin/mr_pipeline.py demo_configs/pipeline_mr/ehr_v1.json

import sys
import os
sys.path.append(
    os.path.join(os.path.dirname(__file__), "../src/python")
)
import pdb
import asyncio
import traceback
import copy
import json
from tqdm import tqdm
from typing import Union, Optional, List, Dict, Coroutine, Callable, Any
from pydantic import BaseModel
from openai import AsyncOpenAI, AsyncAzureOpenAI
from openai import ChatCompletion

from instructionspipe.impl.mapreduce import run_with_configs
from instructionspipe.utils import io_jsonl_write
from instructionspipe.instructions import Instruction, Instructions
from instructionspipe.instructions_runners import InstructionsRunnerBase
from instructionspipe.llm_cli import LlmCli


async def main() -> None:
    configs: Dict = json.loads(open(sys.argv[1], "r").read())
    print(configs)
    in_data_path: str = configs["in_data_path"]
    out_data_path: str = configs["out_data_path"]
    chatml_path: str = configs["chatml_path"]
    if isinstance(configs["pipe"], str):
        configs["pipe"] = json.loads(open(configs["pipe"], "r").read())
    map_conf: Dict = configs["pipe"][0]
    reduce_conf: Dict = configs["pipe"][1]
 
    llm: LlmCli = LlmCli.new_with_configs(configs["llm"])

    # Check
    print("Testing LLM's connection")
    test_resp: Coroutine = llm.async_run("Hi")
    print("Running 'Hi'")
    test_result: str = (await test_resp).choices[0].message.content
    print(test_result)
    print("Testing finished")

    in_samples: List[Dict] = [
        json.loads(x) for x in open(in_data_path, "r").read().split("\n")
        if x not in {""}
    ]
    for in_sample in tqdm(in_samples):
        outputs: Dict = await run_with_configs(
            llm, in_sample, map_conf, reduce_conf 
        )
        in_sample["results"] = outputs
        io_jsonl_write(out_data_path, in_sample, "a")
        io_jsonl_write(chatml_path, outputs["chatmls"], "a")
    return


if __name__ == "__main__":
    asyncio.run(main())
