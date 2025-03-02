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
import os
import torch
import pandas as pd
from pprint import pprint
from datasets import load_dataset
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
    
    data: Optional[List[Dict]] = None
    model: Optional[PreTrainedModel] = None
    tokenizer: Optional[PreTrainedTokenizer] = None
    
    data = [
        json.loads(x) for x in open(configs["in_data_path"], "r").read().split("\n")
        if x not in {""}
    ]
    model, tokenizer = model_and_tokenizer_init(
        configs["model_name_or_path"], 
        configs["tokenizer_name_or_path"],
        configs["is_adapter"]
    )
    device: str = next(model.parameters()).device
    for sample in data:
        msgs: List[Dict] = sample[configs["chatml_col"]]
        if configs["remove_last_msg"]:
            msgs = msgs[:-1]
        if configs["non_system_role"]:
            msgs = chatml_check_and_adjust(msgs, True)
        prompt: str = tokenizer.apply_chat_template(msgs, tokenize=False)
        inputs: BatchEncoding = tokenizer(prompt, return_tensors="pt").to(device)
        outputs = model.generate(inputs.input_ids, max_new_tokens=configs["max_new_tokens"])
        generated_ids: Tensor = outputs[:, inputs.input_ids.shape[1]:]
        gen_text: str = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        sample[configs["gen_text_col"]] = gen_text
    return


if __name__ == "__main__":
    main()
