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


def test_instruction_postproc_ic_ralm() -> None:
    inst: Instruction = instruction_init_naive_case(0)
    examples: List[Document] = documents_init_naive_icl_examples(
        10, "in_text", "out_text" 
    )
    retriever: BaseRetriever = bm25_retriever_init(examples)
    results: List[Document] = retriever.invoke(examples[0].page_content)
    assert(results[0].page_content == examples[0].page_content)
    
    k: int = 3
    assert(inst.examples is None)
    instruction_postproc_ic_ralm(
        inst, 
        examples[0].page_content,
        retriever=retriever,
        k=k,
        embedding=None, 
        search_args={}
    )
    assert(inst.examples is not None)
    assert(len(inst.examples) == 3)
    return

