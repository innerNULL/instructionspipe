# -*- coding: utf-8 -*-
# file: inf.py
# date: 2025-03-02
#
# Usage:
# CUDA_VISIBLE_DEVICES=6 python ./bin/training/inf.py ./bin/training/inf.json

import pdb
import sys
import os
import json
import re
import wandb
import random
import os
import torch
import pandas as pd
from pprint import pprint
from datasets import load_dataset
from tqdm import tqdm
from peft import get_peft_model
from peft import prepare_model_for_kbit_training
from accelerate import infer_auto_device_map
from accelerate import dispatch_model
from typing import Dict, List, Optional, Final, Callable, Union, Tuple
from torch import Tensor
from datasets import Dataset, DatasetDict
from peft import LoraConfig, PeftModel
from peft import AutoPeftModelForCausalLM
from transformers.tokenization_utils_base import BatchEncoding
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers import PreTrainedTokenizer
from transformers import PreTrainedModel
from transformers import BitsAndBytesConfig
from transformers import TrainingArguments
from peft import LoraConfig
from trl import SFTConfig
from trl import SFTTrainer
from trl.trainer.utils import DataCollatorForChatML
from vllm import LLM
from vllm import SamplingParams
from vllm.lora.request import LoRARequest
from accelerate import Accelerator
from accelerate import PartialState


def chatml_check_and_adjust(
    chatml: List[Dict], 
    non_system_role: bool
) -> List[Dict]:
    assert(len(chatml) > 1)
    assert(chatml[0]["role"] in {"system", "user"})
    if non_system_role:
        chatml[0]["role"] = "user"
        if chatml[1]["role"] == "user":
            chatml = [
                chatml[0],
                {"role": "assistant", "content": "Ok."}
            ] + chatml[1:]
    return chatml


def model_resources_init(
    model_name_or_path: str, 
    tokenizer_name_or_path: str,
    adapter_name_or_path: Optional[str]=None,
    max_lora_rank: int=128,
    top_p: float=0.1,
    temperature: float=0.05,
    max_new_tokens: int=1024,
    hf_token: Optional[str]=None
) -> Tuple[LLM, SamplingParams, Optional[LoRARequest]]:
    enable_lora: bool = (adapter_name_or_path is not None)
    model: LLM = LLM(
        model=model_name_or_path,
        tokenizer=tokenizer_name_or_path,
        enable_lora=enable_lora,
        quantization=None,
        seed=2,
        hf_token=hf_token,
        max_lora_rank=max_lora_rank
    )
    sampling_params: SamplingParams = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_new_tokens,
        stop=None
    )
    lora_req: Optional[LoRARequest] = None
    if enable_lora:
        lora_req = LoRARequest(
            adapter_name_or_path, 
            1, 
            adapter_name_or_path
        )
    return model, sampling_params, lora_req


def main() -> None:
    configs: Dict = json.loads(open(sys.argv[1], "r").read())
    print(configs)

    in_data_path: str = configs["in_data_path"]
    model_name_or_path: str = configs["model_name_or_path"]
    tokenizer_name_or_path: str = configs["tokenizer_name_or_path"]
    adapter_name_or_path: str = configs["adapter_name_or_path"]
    max_lora_rank: int = configs["max_lora_rank"]
    temperature: float = configs["temperature"]
    top_p: float = configs["top_p"]
    max_new_tokens: int = configs["max_new_tokens"]
    chatml_col: str = configs["chatml_col"]
    remove_last_msg: bool = configs["remove_last_msg"]
    non_system_role: bool = configs["non_system_role"]
    max_new_tokens: int = configs["max_new_tokens"]
    gen_text_col: str = configs["gen_text_col"]
    gt_text_col: str = configs["gt_text_col"]
    out_data_path: str = configs["out_data_path"]
    extra_cols: List[str] = configs["extra_cols"]

    data: Optional[List[Dict]] = None
    model: Optional[LLM] = None
    sampling_params: Optional[SamplingParams] = None
    lora_req: Optional[LoRARequest] = None
    
    data = [
        json.loads(x) for x in open(in_data_path, "r").read().split("\n")
        if x not in {""}
    ]
    random.seed(2)
    random.shuffle(data)
    model, sampling_params, lora_req = model_resources_init(
        model_name_or_path, 
        tokenizer_name_or_path,
        adapter_name_or_path,
        max_lora_rank=max_lora_rank, 
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens
    )
    outs: List[Dict] = []
    for sample in tqdm(data):
        gen_text: Optional[str] = None
        gt_text: Optional[str] = None
        instruction: Optional[str] = None
        in_text: Optional[str] = None

        msgs: List[Dict] = sample[configs["chatml_col"]]
        if remove_last_msg:
            msgs = msgs[:-1]
            gt_text = sample[configs["chatml_col"]][-1]["content"]     
        if non_system_role:
            msgs = chatml_check_and_adjust(msgs, True)
        
        for msg in msgs:
            if msg["content"] is None:
                msg["content"] = ""

        gen_text: str = model.chat(
            msgs, 
            sampling_params=sampling_params,
            lora_request=lora_req,
            use_tqdm=False
        )[0].outputs[0].text
        sample[gen_text_col] = gen_text
        sample[gt_text_col] = gt_text
        instruction = msgs[0]["content"]
        if len(msgs) > 1:
            in_text = msgs[-1]["content"]
        out: Dict = {
            gen_text_col: gen_text,
            gt_text_col: gt_text,
            "in_text": in_text,
            "instruction": instruction
        }
        for col in extra_cols:
            out[col] = sample[col] 
        outs.append(out)
    
    file = open(out_data_path, "w")
    for out in outs:
        file.write(json.dumps(out, ensure_ascii=False) + "\n")
    file.close()
    print("Inference results are dumped to %s" % out_data_path)
    return


if __name__ == "__main__":
    main()
