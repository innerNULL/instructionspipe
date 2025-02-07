# -*- coding: utf-8 -*-
# file: utils.py
# date: 2024-12-09


import json
from typing import Dict, List, Any


def json2str_kv(json_obj: Dict[str, Any]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for k, v in json_obj.items():
        if isinstance(v, str):
            out[k] = v
        else:
            out[k] = json.dumps(v, ensure_ascii=False, indent=2)
    return out


def io_jsonl_write(
    path: str, 
    data: List[Dict], 
    mode: str="a"
) -> None:
    file = open(path, mode)
    for row in data:
        file.write(
            json.dumps(row, ensure_ascii=False) + "\n"
        )
    file.close()
    return
    
