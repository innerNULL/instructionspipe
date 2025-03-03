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


def model_and_tokenizer_init(
    model_name_or_path: str, 
    tokenizer_name_or_path: str,
    is_adapter: bool=True
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    model: Optional[PreTrainedModel] = None
    if is_adapter:
        model = AutoPeftModelForCausalLM.from_pretrained(
            model_name_or_path,
            quantization_config=None,
            trust_remote_code=True,
            device_map="auto"
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            quantization_config=None,
            trust_remote_code=True,
            device_map="auto"
        )
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name_or_path
    )
    return model, tokenizer


def main() -> None:
    configs: Dict = json.loads(open(sys.argv[1], "r").read())
    print(configs)

    in_data_path: str = configs["in_data_path"]
    model_name_or_path: str = configs["model_name_or_path"]
    tokenizer_name_or_path: str = configs["tokenizer_name_or_path"]
    is_adapter: bool = configs["is_adapter"]
    chatml_col: str = configs["chatml_col"]
    remove_last_msg: bool = configs["remove_last_msg"]
    non_system_role: bool = configs["non_system_role"]
    max_new_tokens: int = configs["max_new_tokens"]
    gen_text_col: str = configs["gen_text_col"]
    gt_text_col: str = configs["gt_text_col"]
    out_data_path: str = configs["out_data_path"]

    data: Optional[List[Dict]] = None
    model: Optional[PreTrainedModel] = None
    tokenizer: Optional[PreTrainedTokenizer] = None
    
    data = [
        json.loads(x) for x in open(in_data_path, "r").read().split("\n")
        if x not in {""}
    ]
    random.shuffle(data)
    model, tokenizer = model_and_tokenizer_init(
        model_name_or_path, 
        tokenizer_name_or_path,
        is_adapter
    )
    device: str = next(model.parameters()).device
    outs: List[Dict] = []
    for sample in tqdm(data):
        gen_text: Optional[str] = None
        gt_text: Optional[str] = None
        instruction: Optional[str] = None
        in_text: Optional[str] = None

        msgs: List[Dict] = sample[configs["chatml_col"]]
        if remove_last_msg:
            msgs = msgs[:-1]
        if non_system_role:
            msgs = chatml_check_and_adjust(msgs, True)

        prompt: str = tokenizer.apply_chat_template(msgs, tokenize=False)
        inputs: BatchEncoding = tokenizer(prompt, return_tensors="pt").to(device)
        outputs = model.generate(inputs.input_ids, max_new_tokens=max_new_tokens)
        generated_ids: Tensor = outputs[:, inputs.input_ids.shape[1]:]
        
        gen_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        gen_text = gen_text.strip("assistant").strip("\n").strip(" ")
        if remove_last_msg:
            gt_text = sample[configs["chatml_col"]][-1]["content"]
        
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
        outs.append(out)
    
    file = open(out_data_path, "w")
    for out in outs:
        file.write(json.dumps(out, ensure_ascii=False) + "\n")
    file.close()
    print("Inference results are dumped to %s" % out_data_path)
    return


if __name__ == "__main__":
    main()
