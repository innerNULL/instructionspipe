# -*- coding: utf-8 -*-
# file: constants.py
# date: 2025-01-02


from typing import Union, Optional, List, Dict, Coroutine, Callable, Any, Set


EMPTY_VAL: str = "N/A"


INVALID_VALS: Set[Optional[str]] = {
    EMPTY_VAL,
    None,
    "",
    " ",
    "NA",
    "N/A",
    "\n"
}
