# -*- coding: utf-8 -*-
# file: quantization.py
# date: 2025-02-23


import os
import sys
import pdb
import json
from typing import Dict, List, Tuple, Optional
from llmcompressor.modifiers.smoothquant import SmoothQuantModifier
from llmcompressor.modifiers.quantization import GPTQModifier
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers import PreTrainedTokenizer
from transformers import PreTrainedModel
from datasets import Dataset
from datasets import DatasetDict
from llmcompressor.transformers import oneshot
from datasets import load_dataset


def dataset_load(
    path_or_name: str, 
    split: Optional[str]=None
) -> Dataset:
    out: Optional[Dataset] = None
    if os.path.exists(path_or_name):
        if path_or_name.split(".")[-1] == "csv":
            out = Dataset.from_pandas(pd.read_csv(path_or_name))
        elif path_or_name.split(".")[-1] == "jsonl":
            out = load_dataset("json", data_files=path_or_name)["train"]
        else:
            raise Exception("Not a supported file format")
    else:
        if split is None:
            raise "Can not loading HuggingFace dataset without split info"
        out = load_dataset(path_or_name, split=split)
    return out


def dataset_load_as_std_clibration_data(
    path_or_name: str,
    target_cols: List[str],
    delimiter: str,
    calibration_sample_size: int,
    max_sequence_len: int,
    tokenizer: PreTrainedTokenizer,
    split: Optional[str]=None,
    seed: int=2
) -> Dataset:
    out: Dataset = dataset_load(path_or_name, split)
    out = out\
        .shuffle(seed=seed)\
        .select(range(calibration_sample_size))

    def preprocess(example: Dict) -> Dict:
        text: str = ""
        for col in target_cols:
            val: str | List[Dict] = example[col]
            if isinstance(val, list) and isinstance(val[0], dict):
                text += tokenizer.apply_chat_template(
                    val, 
                    tokenize=False
                )
            else:
                text += val
            text += delimiter
        return {"text": text}

    def tokenize(sample: Dict) -> Dict:
        return tokenizer(
            sample["text"], 
            padding=False, 
            max_length=max_sequence_len, 
            truncation=True, 
            add_special_tokens=False
        )

    out = out.map(preprocess)
    out = out.map(tokenize, remove_columns=out.column_names)
    return out

def main() -> None:
    configs: Dict = json.loads(open(sys.argv[1], "r").read())
    print(configs)

    model: Optional[PreTrainedModel] = None
    tokenizer: Optional[PreTrainedTokenizer] = None
    data: Otional[Dataset] = None

    model = AutoModelForCausalLM.from_pretrained(
        configs["model_name_or_path"],
        device_map="auto",
        torch_dtype="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(
        configs["model_name_or_path"]
    )
    data = dataset_load_as_std_clibration_data(
        configs["data_name_or_path"], 
        target_cols=configs["target_cols"],
        delimiter=configs["delimiter"],
        calibration_sample_size=configs["calibration_sample_size"],
        max_sequence_len=configs["max_sequence_len"],
        split=configs["data_split"],
        tokenizer=tokenizer
    )
    recipe = [
        SmoothQuantModifier(smoothing_strength=0.8),
        GPTQModifier(scheme="W8A8", targets="Linear", ignore=["lm_head"]),
    ]
    oneshot(
        model=model,
        dataset=data,
        recipe=recipe,
        max_seq_length=configs["max_sequence_len"],
        num_calibration_samples=configs["calibration_sample_size"],
	)
    model.save_pretrained(configs["out_dir"], save_compressed=True)
    tokenizer.save_pretrained(configs["out_dir"])
    print("Successfully dumped to %s" % configs["out_dir"])
    return


if __name__ == "__main__":
    main()
