# -*- coding: utf-8 -*-
# file: etl_sampling_by_key.py
# date: 2025-02-28
#
# Usage:
# python etl_sampling_by_key.py etl_sampling_by_key.json


import sys
import os
import pdb
import json
import random
from typing import Dict, List, Optional, Union


def grouping(data: List[Dict], key: str) -> Dict[str, List[Dict]]:
    out: Dict[str, List[Dict]] = {}
    for row in data:
        group: Union[str, List[Dict]] = row[key]
        group_name: Optional[str] = None
        if isinstance(group, str):
            group_name = group
        elif isinstance(group, list) and isinstance(group[0], dict):
            #print("ChatML key")
            group_name = json.dumps(group, ensure_ascii=False)
        else:
            raise Exception("Unsupported value")
        if group_name not in out:
            out[group_name] = []
        out[group_name].append(row)
    return out


def jsonl_load(path: str) -> List[Dict]:
    return [
        json.loads(x) for x in open(path, "r").read().split("\n")
        if x not in {""}
    ]


def deduplication(data: List[Dict], key: str) -> List[Dict]:
    out: List[Dict] = []
    grouped_data: Dict[str, List[Dict]] = grouping(data, key)
    for k, v in grouped_data.items():
        out.append(v[0])
    return out


def multi_deduplication(data: List[Dict], keys: List[str]) -> List[Dict]:
    for key in keys:
        data = deduplication(data, key)
    return data


def sampling(
    data: List[Dict], 
    key: str, 
    max_group_size: int, 
    seed: int=2
) -> List[Dict]:
    out: List[Dict] = []
    random.seed(seed)
    grouped_data: Dict[str, List[Dict]] = grouping(data, key)
    for k, v in grouped_data.items():
        group_size: int = len(v)
        sample_size: int = min(group_size, max_group_size)
        out.extend(random.sample(v, sample_size))
    return out


def multi_sampling(
    data: List[Dict], 
    keys: List[str], 
    max_group_size: int, 
    seed: int=2
) -> List[Dict]:
    for key in keys:
        data = sampling(data, key, max_group_size, seed)
    return data


def distribution_check(data: List[Dict], key: str) -> None:
    grouped_data: Dict[str, List[Dict]] = grouping(data, key)
    for k, v in sorted(
        [(k, len(v)) for k, v in grouped_data.items()], 
        key=lambda x: x[1],
        reverse=True
    ):
        print("%s: %i" % (k, v))


def jsonl_dump(data: List[Dict], path: str) -> None:
    file = open(path, "w")
    for row in data:
        file.write(json.dumps(row, ensure_ascii=False) + "\n")
    file.close()
    print("Successed dumped data to %s" % path)


def main() -> None:
    configs: Dict = json.loads(open(sys.argv[1], "r").read())
    print(configs)

    # Load data
    data: List[Dict] = jsonl_load(configs["in_data_path"])
    print("Data size: %i" % len(data))

    # Deduplication
    data = multi_deduplication(data, configs["deduplication_keys"])
    print("Data size after deduplication: %i" % len(data))
    
    # Sampling
    data = multi_sampling(
        data, 
        configs["sampling_keys"], 
        configs["max_group_size"],
        2
    )
    print("Data size after sampling: %i" % len(data))
    
    for k in configs["sampling_keys"]:
        print("------ Distribution along '%s' ------" % k)
        distribution_check(data, k)

    jsonl_dump(data, configs["out_data_path"])
    return


if __name__ == "__main__":
    main()
