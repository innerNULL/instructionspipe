# -*- coding: utf-8 -*-
# file: eval_cases_gen.py
# date: 2025-02-04


import sys
import pdb
import json
from typing import Dict, List


DEFAULT_OUT_PATH: str = "./_eval_cases.jsonl"


INPUT_1: str = \
"""
Today I have a meeting on 3:00 pm
""".strip("\n")

INSTRUCTION_1: str = \
"""
Say something about my avalibility on this afternoon
"""

NEG_1: str = \
"""
I have something to do on today's 3:00 P.M.
""".strip("\n")


POS_1_1: str = \
"""
Maybe I will have meeting today.
""".strip("\n")


POS_1_2: str = \
"""
Today Tom has a meeting at afternoon.
""".strip("\n")


INPUT_2: str = \
"""
Current date is 2025-02-03

# Things I did on 2025-02-01
* Cleaning room
* Working on side-project

# Things I did on 2025-02-02
* Shopping
* Having dinner with my parents
""".strip("\n")


INSTRUCTION_2: str = \
"""
Summarize what I did on yesterday as a markdown items list.
"""


NEG_2: str = \
"""
Things I did on yesterday:
* Having dinner with my parents
* Shopping
""".strip("\n")


POS_2_1: str = \
"""
Things I did on yesterday:
* Shopping
* Having dinner with my parents
* Cleaning room
"""


POS_2_2: str = \
"""
I did shopping yesterday, and also had dinner with my parents.
"""


INPUT_3: str = \
"""
# Current Datetime
2025-01-03 19:00:00

# Patient's vital signs
## 2025-01-01 09:10:30
* heart rate: 89
* body temperature: 36

## 2025-01-01 18:20:30
* diastolic pressure: 80
* heart rate: 90
* body temperature: 35

## 2025-01-02 08:40:30
* body temperature: 37
* systolic pressure: 130

## 2025-01-03 15:50:25
* heart rate: 70
""".strip("\n")


INSTRUCTION_3: str = \
"""
Extract vital signs in latest 24 hours as items list.
"""


NEG_3: str = \
"""
Vital signs in latest 24 hours:
* 2025-01-03 15:50:25
  * heart rate: 70
* 2025-01-02 08:40:30
  * body temperature: 37
  * systolic pressure: 130
"""


# `* diastolic pressure: 80` is latest 48 hours result
POS_3_1: str = \
"""
Vital signs in latest 24 hours:
* heart rate: 70
* body temperature: 37
* systolic pressure: 130
* diastolic pressure: 80
"""


POS_3_2: str = \
"""
The requested vital signs are:
* heart rate: 70
* body temperature: 37
* systolic pressure: 130
* diastolic pressure: 80
""".strip("\n")


POS_3_3: str = \
"""
Vital signs on 2025-01-01 09:10:30
* heart rate: 89
* body temperature: 36
""".strip("\n")


POS_3_4: str = \
"""
Here are vital signs in latest 24 hours. heart rate was 70, temperature was 37 and systolic was 130.
""".strip("\n")


INPUT_4: str = \
"""
## Progress Notes
### Progress Note on 2024-12-31

###
""".strip("\n")


CASES: List[Dict] = [
    {"id": 0, "in_text": INPUT_1, "out_text": NEG_1, "instruction": INSTRUCTION_1, "gt_factuality": 1.0, "gt_eligibility": 1.0},
    {"id": 1, "in_text": INPUT_1, "out_text": POS_1_1, "instruction": INSTRUCTION_1, "gt_factuality": 0.0, "gt_eligibility": None},
    {"id": 2, "in_text": INPUT_1, "out_text": POS_1_2, "instruction": INSTRUCTION_1, "gt_factuality": None, "gt_eligibility": 0.0},
    {"id": 3, "in_text": INPUT_2, "out_text": NEG_2, "instruction": INSTRUCTION_2, "gt_factuality": 1.0, "gt_eligibility": 1.0},
    {"id": 4, "in_text": INPUT_2, "out_text": POS_2_1, "instruction": INSTRUCTION_2, "gt_factuality": 0.0, "gt_eligibility": None},
    {"id": 5, "in_text": INPUT_2, "out_text": POS_2_2, "instruction": INSTRUCTION_2, "gt_factuality": 1.0, "gt_eligibility": 0.0},
    {"id": 6, "in_text": INPUT_3, "out_text": NEG_3, "instruction": INSTRUCTION_3, "gt_factuality": 1.0, "gt_eligibility": 1.0},
    {"id": 7, "in_text": INPUT_3, "out_text": POS_3_1, "instruction": INSTRUCTION_3, "gt_factuality": 0.0, "gt_eligibility": 0.0},
    {"id": 8, "in_text": INPUT_3, "out_text": POS_3_2, "instruction": INSTRUCTION_3, "gt_factuality": 0.0, "gt_eligibility": 0.0},
    {"id": 9, "in_text": INPUT_3, "out_text": POS_3_3, "instruction": INSTRUCTION_3, "gt_factuality": None, "gt_eligibility": 0.0},
    {"id": 10, "in_text": INPUT_3, "out_text": POS_3_4, "instruction": INSTRUCTION_3, "gt_factuality": None, "gt_eligibility": 0.0},
]


if __name__ == "__main__":
    out_path: str = DEFAULT_OUT_PATH
    if len(sys.argv) == 2:
        out_path = sys.argv[1]
    out_file = open(out_path, "w")
    for sample in CASES:
        out_file.write(json.dumps(sample) + "\n")
    out_file.close()
    print("Dumped test cases to %s" % out_path)
