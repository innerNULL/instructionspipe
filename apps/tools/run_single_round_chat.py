# -*- coding: utf-8 -*-
# file: run_single_round_chat.py
# date: 2025-04-02


import asyncio
import sys
import os
import pdb
import json
from tqdm import tqdm
from typing import Dict, List, Tuple
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage


def json2chatml(
    data: Dict, 
    msg_configs: List[Dict]
) -> List[Dict]:
    out: List[Dict] = []
    for config in msg_configs:
        role: str = config["role"]
        col: str = config["col"]
        out.append(
            {"role": role, "content": data[col]}
        )
    return out


def batching(samples: List[Dict], batch_size: int=8) -> List[List[Dict]]:
    out: List[List[Dict]] = []
    batch: List[Dict] = []
    for sample in samples:
        batch.append(sample)
        if len(batch) == batch_size:
            out.append(batch)
            batch = []
    if len(batch) > 0:
        out.append(batch)
    return out


async def main() -> None:
    configs: Dict = json.loads(open(sys.argv[1], "r").read())
    print(configs)

    in_data_path: str = configs["in_data_path"]
    out_data_path: str = configs["out_data_path"]
    out_col: str = configs["out_col"]
    llm_configs: Dict = configs["llm"]
    chatml_configs: Dict = configs["chatml"]

    model: ChatOpenAI = ChatOpenAI(
        model=llm_configs["model"],
        temperature=llm_configs["temperature"],
        max_tokens=llm_configs["max_tokens"],
        api_key=llm_configs["api_key"],
        base_url=llm_configs["base_url"]
    )

    samples: List[Dict] = [
        json.loads(x) for x in open(in_data_path, "r").read().split("\n")
        if x not in {""}
    ]
    out_file = open(out_data_path, "w")
    for batch in tqdm(batching(samples)):
        msgs: List[List[Dict]] = [
            json2chatml(x, chatml_configs) for x in batch
        ]
        resps: List[AIMessage] = await model.abatch(msgs)
        results: List[Dict] = []
        for i, sample in enumerate(batch):
            sample[out_col] = resps[i].content
            out_file.write(json.dumps(sample, ensure_ascii=False) + "\n")
    out_file.close()
    print("Dumped the results to %s" % out_data_path)
    return 


if __name__ == "__main__":
    asyncio.run(main())
