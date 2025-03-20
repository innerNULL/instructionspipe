# -*- coding: utf-8 -*-
# file: mr_pipeline.py
# date: 2024-11-09
#
# Run example:
# python ./inf.py ./demo_configs/mapreduce_summ/ehr.json
# python ./inf.py ./demo_configs/mapreduce_summ/ehr_v1.json


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


async def inf_with_configs(configs: Dict, append_mode: bool=True) -> None:
    print("Running inference with configs:\n{}".format(configs))
    in_data_path: str = configs["in_data_path"]
    out_data_path: str = configs["out_data_path"]
    chatml_path: str = configs["chatml_path"]
    chatml_meta_path: str = chatml_path + ".meta.jsonl"
    if isinstance(configs["pipe"], str):
        configs["pipe"] = json.loads(open(configs["pipe"], "r").read())
    map_conf: Dict = configs["pipe"][0]
    reduce_conf: Dict = configs["pipe"][1]
    
    if not append_mode:
        try:
            assert(not os.path.exists(out_data_path))
            assert(not os.path.exists(chatml_path))
            assert(not os.path.exists(chatml_meta_path))
        except Exception as e:
            print("One of following path exists, pls remove or rename them:")
            print("* %s" % out_data_path)
            print("* %s" % chatml_path)
            print("* %s" % chatml_meta_path)
            raise e

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
        try:
            outputs: Dict = await run_with_configs(
                llm, in_sample, map_conf, reduce_conf 
            )
        except Exception as e:
            print(traceback.format_exc())
            continue
        in_sample["results"] = outputs
        io_jsonl_write(out_data_path, [outputs], "a")
        io_jsonl_write(chatml_path, outputs["chatmls"], "a")

        in_sample["session_id"] = outputs["chatmls"][0]["session_id"]
        io_jsonl_write(chatml_meta_path, [in_sample], "a")
    return


async def main() -> None:
    configs_path: str = sys.argv[1]
    configs: List[Dict] = []
    if os.path.isdir(configs_path):
        print("Using all JSONs as configuration under directory %s" % configs_path)
        for file in os.listdir(configs_path):
            full_path: str = os.path.join(configs_path, file)
            if full_path[-5:] != ".json":
                print("Skip %s" % full_path)
            config: Dict = json.loads(open(full_path, "r").read())
            configs.append(config)
    else:
        config: Dict = json.loads(open(configs_path, "r").read())
        configs.append(config)
    
    for i, conf in enumerate(tqdm(configs)):
        await inf_with_configs(conf, i != 0)


if __name__ == "__main__":
    asyncio.run(main())
