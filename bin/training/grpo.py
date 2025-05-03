# -*- coding: utf-8 -*-
# file: grpo.py
# date: 2025-04-26
"""
## Run
CUDA_VISIBLE_DEVICES=1,4 python bin/training/grpo.py bin/training/grpo.json
"""

import pdb
import os
import sys
import asyncio
import json
import torch
import wandb
import importlib
from typing import (
    Dict, 
    List, 
    Optional,
    Set,
    Tuple,
    Any
)
from trl import (
    GRPOConfig, 
    GRPOTrainer
)
from torch import Tensor
from datasets import (
    Dataset, 
    DatasetDict
)
from peft import (
    LoraConfig, 
    PeftModel
)
from transformers.tokenization_utils_base import BatchEncoding
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers import PreTrainedTokenizer
from transformers import PreTrainedModel
from transformers import BitsAndBytesConfig
from transformers import TrainingArguments
from trl.trainer.utils import DataCollatorForChatML
from accelerate import Accelerator
from accelerate import PartialState
from datasets import load_dataset
from peft import (
    get_peft_model,
    prepare_model_for_kbit_training
)


INVALID_VALS: Set = {
    None, 
    "", 
    "\n",
    " "
}


def chatml_check_and_adjust(
    chatml: List[Dict], 
    model: str,
    remove_null: bool=True
) -> List[Dict]:
    assert(len(chatml) > 1)
    assert(chatml[0]["role"] in {"system", "user"})
    user_only_llms: List[str] = ["mistral", "gemma"]
    for pattern in user_only_llms:
        if pattern in model.lower():
            chatml[0]["role"] = "user"
            if chatml[1]["role"] == "user":
                chatml = [
                    chatml[0],
                    {"role": "assistant", "content": "Ok."}
                ] + chatml[1:]
            break
    if remove_null:
        for msg in chatml:
            if msg["content"] is None:
                msg["content"] = ""
    return chatml


def dataset_chatml_adj(
    datasets: Dict[str, Dataset],
    chatml_col: str, 
    model: str
) -> Dict[str, Dataset]:
    def chatml_adj(inputs: Dict) -> Dict:
        inputs[chatml_col] = chatml_check_and_adjust(
            inputs[chatml_col], model
        )
        if inputs[chatml_col][-1]["role"] != "user":
            inputs[chatml_col] = inputs[chatml_col][:-1]
        return inputs
    for k, v in datasets.items():
        v = v.map(chatml_adj)
        datasets[k] = v.rename_column(chatml_col, 'prompt')              
    return datasets


def dataset_load(
    path_or_name: str,  
    split: Optional[str]=None
) -> Dataset:
    if os.path.exists(path_or_name):
        if path_or_name.split(".")[-1] == "csv":
            return Dataset.from_pandas(pd.read_csv(path_or_name))
        elif path_or_name.split(".")[-1] == "jsonl":
            return load_dataset("json", data_files=path_or_name)["train"]
        else:
            raise Exception("Not a supported file format")
    else:
        if split is None:
            raise "Can not loading HuggingFace dataset without split info"
        return load_dataset(path_or_name, split=split)


def datasets_load(
    train_path_or_name: str,
    val_path_or_name: str,
    train_size: int, 
    val_size: int, 
    train_split: Optional[str]=None,
    val_split: Optional[str]=None,
    seed: int=2
) -> Dict[str, Dataset]:
    train: Dataset = dataset_load(train_path_or_name, train_split)
    val: Dataset = dataset_load(val_path_or_name, val_split)
    if len(train) > train_size:
        train = train.shuffle(seed=seed).select(range(train_size))
    if len(val) > val_size:
        val = val.shuffle(seed=seed).select(range(val_size))
    return {"train": train, "val": val}


def configs_load(path: str) -> Dict:
    configs: Dict = json.loads(open(path, "r").read())
    if configs["hf"]["token"] in INVALID_VALS:
        configs["hf"]["token"] = os.environ.get(
            "HUGGING_FACE_HUB_TOKEN", None
        )
    if configs["hf"]["token"] in INVALID_VALS:      
        configs["hf"]["token"] = os.environ.get("HF_TOKEN", None)
    if configs["wandb"]["key"] in INVALID_VALS:
        configs["wandb"]["key"] = os.environ.get("WANDB_API_KEY", None)
    if configs["wandb"]["project"] in INVALID_VALS:
        configs["wandb"]["project"] = os.environ.get("WANDB_PROJECT", None)  
    print(configs)           
    return configs


def wandb_init(key: str, project: str) -> None:
    wandb.login(key=key)
    wandb.init(project=project)


def model_and_tokenizer_init(
    model_name_or_path: str, 
    tokenizer_name_or_path: str,
    adapter_conf: Dict={"type": None},
    pad_token: str="<|finetune_right_pad|>",
    hf_token: Optional[str]=None
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    user_peft: bool = (adapter_conf["type"] is not None)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        #quantization_config=bnb_config if user_peft else None,
        trust_remote_code=True,
        device_map="auto",
        token=hf_token
    )
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name_or_path,
        token=hf_token
    )
    if tokenizer.pad_token is None: 
        if tokenizer.unk_token is not None:
            tokenizer.add_special_tokens({'pad_token': tokenizer.unk_token})
            model.resize_token_embeddings(len(tokenizer))
            print("Using `tokenizer.unk_token` as `tokenizer.pad_token`")
    if tokenizer.pad_token is None:
        #tokenizer.pad_token = pad_token
        tokenizer.add_special_tokens({'pad_token': pad_token})
        model.resize_token_embeddings(len(tokenizer))
        print("Using '%s' as `tokenizer.pad_token`" % pad_token)
    assert(tokenizer.pad_token != tokenizer.eos_token)
    peft_config = None
    if adapter_conf["type"] is not None:
        if adapter_conf["type"] == "lora":
            peft_config = LoraConfig(
                r=adapter_conf["lora_rank"], # LoRA rank
                lora_alpha=adapter_conf["lora_alpha"], # Scaling factor
                target_modules=["q_proj", "v_proj"], # Target specific modules to apply LoRA
                lora_dropout=adapter_conf["lora_dropout"],
                bias="none",
                task_type="CAUSAL_LM",
            )
    if peft_config is not None:
        model = get_peft_model(model, peft_config)
    return model, tokenizer


async def reward_model_load(configs: Dict) -> Any:
    plugin_path: str = configs["plugin_path"]
    plugin_dir: str = os.path.dirname(plugin_path)
    plugin_name: str = os.path.basename(plugin_path).strip(".py")
    sys.path.append(plugin_dir)
    plugin = importlib.import_module(plugin_name)
    Reward = plugin.Reward
    rm: Reward = Reward.new_with_configs(configs)
    await plugin.test(configs)
    return rm


async def main() -> None:
    configs: Dict = configs_load(sys.argv[1])
    hf_conf: Dict = configs["hf"]
    wandb_conf: Dict = configs["wandb"]
    model_conf: Dict = configs["model"]
    data_conf: Dict = configs["data"]
    train_conf: Dict = configs["train"]
    peft_conf: Dict = configs["peft"]
    reward_conf: Dict = configs["reward"]

    wandb_init(wandb_conf["key"], wandb_conf["project"])
    datasets: Optional[Dict[str, Dataset]] = None
    model: Optional[PreTrainedModel] = None
    tokenizer: Optional[PreTrainedTokenizer] = None
    collator: Optional[DataCollatorForChatML] = None
    trainer: Optional[GRPOTrainer] = None
    train_config: Optional[GRPOConfig] = None
    reward_model: Optional[Any] = None

    reward_model: Any = await reward_model_load(reward_conf)
    datasets = datasets_load(
        data_conf["train_data_path"],
        data_conf["val_data_path"],
        data_conf["train_size"],
        data_conf["val_size"]
    )
    datasets = dataset_chatml_adj(
        datasets, 
        data_conf["chatml_col"], 
        model_conf["model_name_or_path"]
    )
    model, tokenizer = model_and_tokenizer_init(
        model_conf["model_name_or_path"], 
        model_conf["tokenizer_name_or_path"],
        adapter_conf=peft_conf,
        hf_token=hf_conf["token"]
    )
    train_config = GRPOConfig(
        overwrite_output_dir=False,
        per_device_train_batch_size=train_conf["per_device_train_batch_size"],
        per_device_eval_batch_size=train_conf["per_device_eval_batch_size"],
        gradient_accumulation_steps=train_conf["gradient_accumulation_steps"],
        max_prompt_length=train_conf["max_length"],
        max_completion_length=train_conf["max_length"],
        optim="paged_adamw_32bit",
        logging_steps=1,
        learning_rate=train_conf["learning_rate"],
        fp16=model_conf["quantization"],
        max_grad_norm=0.3,
        num_train_epochs=train_conf["num_epochs"],
        save_strategy="steps",
        evaluation_strategy="steps",
        save_steps=0.05,
        eval_steps=0.05,
        warmup_ratio=0.05,
        save_total_limit=3,
        report_to="wandb",  
        save_safetensors=False,
        lr_scheduler_type="cosine",
        seed=42,
        load_best_model_at_end=True,
        push_to_hub=False,
        output_dir=train_conf["out_dir"],
        #dataset_kwargs={"skip_prepare_dataset": True},
        remove_unused_columns=False,
        #packing=True,
        eval_on_start=True,
        group_by_length=False, # Sad but not sure how to use this
        ## GRPO
        temperature=0.9,
        top_p=1.0,
        top_k=50,
        repetition_penalty=1.0,
        beta=0.04,
        num_iterations=1,
        epsilon=0.2,
        num_generations=2
    )
    trainer = GRPOTrainer(
        model,
        train_dataset=datasets["train"],
        eval_dataset=datasets["val"],
        args=train_config,
        processing_class=tokenizer,
        reward_funcs=reward_model.run,
    )
    trainer.train()
    return


if __name__ == "__main__":
    asyncio.run(main())

