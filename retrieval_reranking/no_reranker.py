import argparse
import json
import pandas as pd
import nltk
from rank_bm25 import BM25Okapi
import numpy as np
import os
import sys
import torch
import tqdm
from transformers import BloomTokenizerFast, BloomModel, BloomForCausalLM
from accelerate import Accelerator


parser = argparse.ArgumentParser(description='A tiny script that does not do any reranking, but simply spits out BM25 top-3N as QA pairs..')
parser.add_argument('--averitec_file', default="/rds/user/mss84/hpc-work/datasets/averitec/initial_test_dataset/dev.neat.top_100_with_questions.recombined.json", help='')
parser.add_argument('--n', default=3, help='')
args = parser.parse_args()

with open(args.averitec_file) as f:
    examples = json.load(f)

for example in tqdm.tqdm(examples):
    top_n = example["bm25_qas"][:args.n] if "bm25_qas" in example else []
    pass_through = [{"question": qa[0], "answers": [{"answer": qa[1]}]} for qa in top_n]

    example["questions"] = pass_through

print(json.dumps(examples, indent=4))