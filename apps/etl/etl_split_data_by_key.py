# -*- coding: utf-8 -*-
# file: etl_instruction_training_data.py
# date: 2025-02-19
#
# Usage:
# python ./bin/etl/etl_instruction_training_data.py ./bin/etl/etl_instruction_training_data.json


import os
import sys
import pdb
import json
import random
from typing import Dict, List


def data_grouping(data: List[Dict], key: str) -> Dict[str, List[Dict]]:
    out: Dict[str, List[Dict]] = {}
    for sample in data:
        key_name: str = sample[key]
        if key_name not in out:
            out[key_name] = []
        out[key_name].append(sample)
    return out


def grouped_data_splitting(
    grouped_data: Dict[str, List[Dict]],
    train_val_test_ratio: List[float],
    seed: int=2
) -> Dict[str, List[Dict]]:
    random.seed(seed)
    keys: List[str] = list(grouped_data.keys())
    random.shuffle(keys)
    size: int = len(keys)
    split1: int = int(size * train_val_test_ratio[0])
    split2: int = split1 + int(size * train_val_test_ratio[1])
    train_keys: List[str] = keys[:split1]
    val_keys: List[str] = keys[split1:split2]
    test_keys: List[str] = keys[split2:]
    train_data: List[Dict] = []
    val_data: List[Dict] = []
    test_data: List[Dict] = []
    for k, v in grouped_data.items():
        if k in train_keys:
            train_data.extend(v)
        if k in val_keys:
            val_data.extend(v)
        if k in test_keys:
            test_data.extend(v)
    return {
        "train": train_data, 
        "val": val_data, 
        "test": test_data
    }


def main() -> None:
    configs: Dict = json.loads(open(sys.argv[1], "r").read())
    print(configs)

    random.seed(configs["seed"])

    train_val_test_ratio: List[float] = configs["train_val_test_ratio"]
    in_data_path: str = configs["in_data_path"]
    out_data_dir: str = configs["out_data_dir"]
    instruction_name_col: str = configs["instruction_name_col"]

    data: List[Dict] = [
        json.loads(x) for x in open(in_data_path, "r").read().split("\n")
        if x not in {""}
    ]
    grouped_data: Dict[str, List[Dict]] = data_grouping(data, instruction_name_col)
    datasets: Dict[str, List[Dict]] = grouped_data_splitting(
        grouped_data, train_val_test_ratio
    )

    os.system("mkdir -p %s" % out_data_dir)
    for k, v in datasets.items():
        path: str = os.path.join(out_data_dir, k + ".jsonl")
        file = open(path, "w")
        for sample in v:
            file.write(json.dumps(sample, ensure_ascii=False) + "\n")
        file.close()
    print("Datasets are dumped to %s" % out_data_dir)
    return


if __name__ == "__main__":
    main()

