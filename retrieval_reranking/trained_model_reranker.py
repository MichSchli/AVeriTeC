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
from transformers import BertTokenizer, BertForSequenceClassification

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from models.DualEncoderModule import DualEncoderModule

parser = argparse.ArgumentParser(description='A script that reranks by relying on a trained model to score claim-question-answer triples.')
parser.add_argument('--averitec_file', default="/rds/user/mss84/hpc-work/datasets/averitec/full_data/date.test.with_qs_combined.json", help='')
parser.add_argument('--n', default=3, help='')
args = parser.parse_args()

with open(args.averitec_file) as f:
    examples = json.load(f)

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2, problem_type="single_label_classification") # Must specify single_label for some reason
best_checkpoint = "/rds/user/mss84/hpc-work/checkpoint_files/averitec/bert_dual_encoder_true_withfcb-epoch=19-val_loss=0.00-val_acc=0.93.ckpt"
device = "cuda:0" if torch.cuda.is_available() else "cpu"
trained_model = DualEncoderModule.load_from_checkpoint(best_checkpoint, tokenizer = tokenizer, model = bert_model).to(device)

def triple_to_string(x):
    return " </s> ".join([item.strip() for item in x])

for example in tqdm.tqdm(examples):
    strs_to_score = []
    values = []

    bm25_qas = example["bm25_qas"] if "bm25_qas" in example else []

    for question,answer, source in bm25_qas:
        claim = example["claim"]

        str_to_score = triple_to_string([claim, question, answer])

        strs_to_score.append(str_to_score)
        values.append([question, answer, source])

    if len(bm25_qas) > 0:
        encoded_dict = tokenizer(
            strs_to_score,
            max_length=512,
            padding="longest",
            truncation=True,
            return_tensors="pt"
        ).to(device)

        input_ids = encoded_dict['input_ids']
        attention_masks = encoded_dict['attention_mask']

        scores = torch.softmax(trained_model(input_ids, attention_mask=attention_masks).logits, axis=-1)[:, 1]
    
        top_n = torch.argsort(scores, descending=True)[:args.n]
        pass_through = [{"question": values[i][0], "answers": [{"answer": values[i][1], "source_url": values[i][2]}]} for i in top_n]
    else:
        pass_through = []    

    example["questions"] = pass_through

print(json.dumps(examples, indent=4))