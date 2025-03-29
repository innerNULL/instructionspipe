# -*- coding: utf-8 -*-
# file: instructions_processor.py
# date: 2025-03-24


import pdb
from typing import Union, Optional, List, Dict, Coroutine, Callable, Any, Set
from pydantic import BaseModel
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStore

from .instructions import Instruction
from .instructions import Instructions
from .instructions import instruction_postproc_ic_ralm


class IcRaLmConfig(BaseModel):
    k: int = 1


class IcRaLmConfigs(BaseModel):
    example_input_col: str
    example_output_col: str
    keys: List[str]
    instruction_configs: Dict[str, IcRaLmConfig]


class InstructionsProcessor:
    def __init__(self):
        self.examples_retriever: Optional[BaseRetriever | Dict[str, BaseRetriever]] = None
        self.knowledge_retriever: Optional[BaseRetriever] = None
        self.icralm_configs: Optional[IcRaLmConfigs] = None

    @classmethod
    def new(cls,
        icralm_configs: Dict | IcRaLmConfigs,
        examples_retriever: BaseRetriever | Dict[str, BaseRetriever]
    ):
        out = cls()
        if isinstance(icralm_configs, dict):
            icralm_configs = IcRaLmConfigs.parse_obj(icralm_configs)

        out.examples_retriever = examples_retriever
        out.icralm_configs = icralm_configs
        return out

    def run_icralm_preprocessing(self, 
        instructions: Instructions,
        input_data: str | Dict[str, str]
    ) -> Instructions:
        for instruction in instructions.instructions:
            name: str = instruction.name
            if name not in self.icralm_configs.instruction_configs:
                continue
            instruction = instruction_postproc_ic_ralm(
                instruction, 
                input_data, 
                retriever=self.examples_retriever,
                keys=self.icralm_configs.keys,
                k=self.icralm_configs.instruction_configs[name].k,
                example_input_col=self.icralm_configs.example_input_col,
                example_output_col=self.icralm_configs.example_output_col
            )
        return instructions


