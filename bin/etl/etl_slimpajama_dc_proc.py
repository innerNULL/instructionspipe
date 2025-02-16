# -*- coding: utf-8 -*- 
# file: text_corpus_slimpajama_dc_processor.py
# date: 2024-06-24
#
# This data filtering refers to paper https://arxiv.org/abs/2309.10818
# It mainly using MinHash + LSH to do deduplication, also filter corpus
# which length is too short.
#
# Usage:
# python text_corpus_slimpajama_dc_processor.py ${config_path}
#
# Configs:
# workspace: Path to cache some intermediate files.
# data_path_or_name: Can be JSON lines data path or HF datasets' name
# target_text_cols: The target text column name with which we will do 
#     data filtering.
# n_gram: The N-gram with which we build vocabulary for MinHash
# lsh_threshold: Jaccard distance threshold, higher than which we will 
#     believe there are some duplications
# most_dup: Most duplications we can accept, if this sample has more 
#     then `most_dup` duplications, then it will be filtered.
# low_length_filter: The minimum length filter, if any fields under it 
#     has less than `{value}` words, the sample will be filtered
# output_path: Processed JSON lines output path.
# debug: Open debug mode, if opened, the duplication cases will be saved 
#     in `dbg_corpus` 


import pdb
import sys
import os
import string
import json
from functools import partial
from typing import List, Dict, Callable, Optional
from sklearn.feature_extraction.text import CountVectorizer 
from datasketch import MinHash
from datasketch import MinHashLSH


class CharNGramExtractor:
    def __init__(self, n_for_n_gram: int):
        self.n_for_n_gram: int = n_for_n_gram
        self.engine: CountVectorizer = CountVectorizer(
            ngram_range=(n_for_n_gram, n_for_n_gram), analyzer="char"
        )
        self.analyzer: partial = self.engine.build_analyzer()
    
    def text_preprocessor(self, text: str) -> str:
        return text.translate(
            str.maketrans('', '', string.punctuation)
        )

    def run(self, text: str) -> List[str]:
        return self.analyzer(self.text_preprocessor(text))


class MinHashLshManagement:
    def __init__(self, 
        text_fields: List[str], 
        threshold=0.75, 
        num_perm=128
    ):
        self.num_perm: int = num_perm
        self.lshs: Dict[str, MinHashLSH] = {}
        for text_field in text_fields:
            self.lshs[text_field] = MinHashLSH(threshold=threshold, num_perm=num_perm)

    def insert_lsh(self, field: str, text_id: str, minhash: MinHash) -> None:
        self.lshs[field].insert(text_id, minhash)

    def insert_n_grams(self, field: str, text_id: str, n_grams: List[str]) -> None:
        min_hash: MinHash = MinHash(num_perm=self.num_perm)
        for n_gram in n_grams:
            min_hash.update(n_gram.encode('utf8'))
        self.insert_lsh(field, text_id, min_hash)

    def query_with_lsh(self, field: str, minhash: MinHash):
        return self.lshs[field].query(minhash)

    def query_with_n_grams(self, field: str, n_grams: List[str]):
        min_hash: MinHash = MinHash(num_perm=self.num_perm)
        for n_gram in n_grams:
            min_hash.update(n_gram.encode('utf8'))
        return self.query_with_lsh(field, min_hash)


def test() -> None:
    minhash_permutation_num: int = 128
    n_for_n_gram: int = 7
    n_gram_runner: CharNGramExtractor = CharNGramExtractor(n_for_n_gram)
    lshs: MinHashLshManagement = MinHashLshManagement(["test"])
    case1: str = "minhash is a probabilistic data structure for estimating the similarity between datasets;" 
    case2: str = "minhash is a probability data structure for estimating the similarity between documents."
    case1_n_grams: List[str] = n_gram_runner.run(case1)
    case2_n_grams: List[str] = n_gram_runner.run(case2)
    lshs.insert_n_grams("test", "1", case1_n_grams)
    lshs.insert_n_grams("test", "2", case2_n_grams)
    result = lshs.query_with_n_grams("test", case1_n_grams)
    assert("1" in result)


def main() -> None:
    configs: Dict = json.loads(open(sys.argv[1], "r").read())
    print(configs)

    minhash_permutation_num: int = configs["minhash_permutation_num"]
    n_for_n_gram: int = configs["n_gram"]
    
    n_gram_runner: CharNGramExtractor = CharNGramExtractor(n_for_n_gram)
    lshs: MinHashLshManagement = MinHashLshManagement(
        configs["target_text_cols"], configs["lsh_threshold"], 128
    )
    
    dbg_corpus: Dict[str, str] = {}
    data_file = open(configs["data_path_or_name"], "r")
    output_file = open(configs["output_path"], "w")
    record: str = data_file.readline()
    sample_id: int = 1
    filtered_cnt: int = 0
    while record:
        sample: Dict = json.loads(record)
        recorder: Dict[str, bool] = {}
        remove: bool = False
        for text_col in configs["target_text_cols"]:
            text: str = sample[text_col]
            text_id: str = "{}-{}".format(sample_id, text_col)
            
            print("Checking corpus {}".format(text_id))
            # Low length filter
            if not remove:
                word_cnt: int = len(text.replace("\n", " ").split(" "))
                if word_cnt < configs["low_length_filter"][text_col]:
                    print("Low length filter triggered for {}".format(text_id))
                    remove = True
            
            # Deduplication
            if not remove:
                n_grams: List[str] = n_gram_runner.run(text)
                duplications: List[str] = lshs.query_with_n_grams(text_col, n_grams)
                if len(duplications) <= configs["most_dup"]:
                    lshs.insert_n_grams(text_col, text_id, n_grams)
                    if configs["debug"]:
                        #print("Cache corpus {} for dbg purpose".format(text_id))
                        dbg_corpus[text_id] = text
                else:
                    print("Corpus {} is a duplication of: {}".format(text_id, ", ".join(duplications)))
                    pdb.set_trace()
                    remove = True

        record = data_file.readline()
        sample_id += 1
        filtered_cnt += int(remove)
        if not remove:
            sample["sample_id"] = sample_id
            output_file.write(json.dumps(sample, ensure_ascii=False) + "\n")
        print("Filtering ratio: {}".format(filtered_cnt / sample_id))

    data_file.close()
    output_file.close()
    print("Output dumped to %s" % configs["output_path"])


if __name__ == "__main__":
    test()
    main()
