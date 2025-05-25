# -*- coding: utf-8 -*-
# file: cost_estimator_inf.py
# date: 2025-02-11
#
# Usage:
# python ./bin/cost_estimator_inf.py ./bin/cost_estimator_inf.json


import sys
import os
import pdb
import json
from tqdm import tqdm
from typing import Dict, List, Optional
from transformers import AutoTokenizer
from transformers import PreTrainedTokenizerBase
from transformers.tokenization_utils_base import BatchEncoding


def encoding_text_extraction(in_data: Dict, target_cols: List[str]) -> str:
     out: str = ""
     for col in target_cols:
         val: Union[str, List[Dict]] = in_data[col]
         if isinstance(val, str):
             out += val
         elif isinstance(val, list) and isinstance(val[0], dict):
             for i in range(0, len(val) - 1):
                 out += val[i]["role"] + ":\n"
                 out += val[i]["content"]
                 out += "\n\n"
         out += "\n\n"
     return out.strip("\n")


def main() -> None:
    configs: Dict = json.loads(open(sys.argv[1], "r").read())
    in_data_path: str = configs["in_data_path"]
    encoding_cols: List[str] = configs["encoding_cols"]
    encoding_price_per_1m: float = configs["encoding_price_per_1m"]
    decoding_price_per_1m: float = configs["decoding_price_per_1m"]
    inf_sample_size: int = configs["inf_sample_size"]
    
    in_data: List[Dict] = [
        json.loads(x) for x in open(in_data_path, "r").read().split("\n")
        if x not in {""}
    ][:configs["max_sample_size"]]
    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=configs["tokenizer"]
    )

    in_tokens_nums: List[float] = []
    out_tokens_nums: List[float] = []
    for sample in tqdm(in_data):
        encoding_txt: str = encoding_text_extraction(
            sample, encoding_cols
        )
        encoding: BatchEncoding = tokenizer(
            encoding_txt,
            add_special_tokens=True,
            truncation=False,
            return_tensors="np"
        )
        in_tokens_num: float = float(encoding.input_ids.shape[-1])
        out_tokens_num: float = configs["io_length_ratio"] * in_tokens_num
        in_tokens_nums.append(in_tokens_num)
        out_tokens_nums.append(out_tokens_num)
    
    avg_encoding_tokens: float = sum(in_tokens_nums) / len(in_tokens_nums)
    avg_decoding_tokens: float = sum(out_tokens_nums) / len(out_tokens_nums)
    total_encoding_tokens: float = inf_sample_size * avg_encoding_tokens
    total_decoding_tokens: float = inf_sample_size * avg_decoding_tokens
    estimzted_encoding_cost: float = total_encoding_tokens / 1000000.0 * encoding_price_per_1m
    estimzted_decoding_cost: float = total_decoding_tokens / 1000000.0 * decoding_price_per_1m
    print("Avg encoding tokens num: %s" % avg_encoding_tokens)
    print("Avg decoding tokens num: %s" % avg_decoding_tokens)
    print("Estimated encoding cost: %s" % estimzted_encoding_cost)
    print("Estimated decoding cost: %s" % estimzted_decoding_cost)
    return None


if __name__ == "__main__":
    main()
