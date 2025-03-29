# -*- coding: utf-8 -*-
# file: utils.py
# date: 2025-03-24


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


def instruction_init_naive_case(case_id: int=0) -> Instruction:
    out = Instruction(
        name="instruction_case_%i" % case_id,
        input_desc="Input descripiont of the instruction case %i" % case_id
    ) 
    return out


def documents_init_naive_icl_examples(
    num: int, 
    in_text_col: str, 
    out_text_col: str
) -> List[Document]:
    out: List[Document] = []
    for i in range(num):
        in_text: str = "Input text for ICL example %i." % i
        out_text: str = "Output text for ICL example %i." % i
        doc: Document = Document(
            page_content=in_text,
            metadata={
                "source": "doc-%i" % i,
                in_text_col: in_text, 
                out_text_col: out_text
            }
        )
        out.append(doc)
    return out


def bm25_retriever_init(docs: List[Document]) -> BaseRetriever:
    return BM25Retriever.from_documents(docs)


def init_jsonl(num: int, col_num: int) -> List[Dict]:
    out: List[Dict] = []
    for i in range(num):
        curr_out: Dict = {}
        for j in range(col_num):
            curr_out["col%i" % j] = "This content of column %i in sample %i" % (j, i)
        out.append(curr_out)
    return out


def init_instructions(size: int) -> Instructions:
    instructions: List[Instruction] = []
    for i in range(size):
        curr: Instruction = Instruction(
            name="instruction-%i" % i,
            input_desc="The input of instruction %i" % i,
            output_desc="The output of instruction %i" % i,
        )
        instructions.append(curr)
    return Instructions(
        instructions=instructions,
        result=None, 
        finished=False
    )

