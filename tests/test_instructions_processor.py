# -*- coding: utf-8 -*-
# file: test_instructions.py
# date: 2025-03-24
#
# Usage:
# python -m pytest ./tests/test_instructions.py


import pdb
from typing import Dict, List, Optional
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStore

from instructionspipe.instructions import Instruction
from instructionspipe.instructions import Instructions
from instructionspipe.instructions_processor import InstructionsProcessor
from instructionspipe.instructions import instruction_postproc_ic_ralm

import utils


def test_instruction_postproc_ic_ralm() -> None:
    inst: Instruction = utils.instruction_init_naive_case(0)
    examples: List[Document] = utils.documents_init_naive_icl_examples(
        10, "in_text", "out_text" 
    )
    retriever: BaseRetriever = utils.bm25_retriever_init(examples)
    icralm_configs: Dict = {
        "example_input_col": "in_text",
        "example_output_col": "out_text",
        "keys": ["col0", "col2"],
        "instruction_configs": {
            "instruction-0": {"k": 1},
            "instruction-1": {"k": 2}
        }
    }
    instructions: Instructions = utils.init_instructions(8)
    processor: InstructionsProcessor = InstructionsProcessor.new(
        examples_retriever=retriever,
        icralm_configs=icralm_configs
    )
    input_data: List[Dict] = utils.init_jsonl(10, 5)
    processor.run_icralm_preprocessing(
        instructions=instructions, 
        input_data=input_data[0]
    )
    for instruction in instructions.instructions:
        name: str = instruction.name
        if name not in icralm_configs["instruction_configs"]:
            continue
        examples: List = instruction.examples
        assert(len(examples) == icralm_configs["instruction_configs"][name]["k"])
    return

