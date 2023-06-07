import random
from torch.utils.data import DataLoader, TensorDataset, random_split, RandomSampler, Dataset
import pandas as pd
import numpy as np
import torch
import pytorch_lightning as pl
import json
from nltk.tokenize import word_tokenize
import tqdm

class JustificationProductionDataLoader(pl.LightningDataModule):
  def __init__(self, tokenizer, batch_size):
    super().__init__()
    self.tokenizer = tokenizer
    self.batch_size = batch_size

  def tokenize_strings(self, source_sentences, max_length=512, pad_to_max_length=False, return_tensors="pt"):
    encoded_dict = self.tokenizer(
            source_sentences,
            max_length=max_length,
            padding="max_length" if pad_to_max_length else "longest",
            truncation=True,
            return_tensors=return_tensors
        )

    input_ids = encoded_dict['input_ids']
    attention_masks = encoded_dict['attention_mask']

    return input_ids, attention_masks

  def encode_sentences(self, source_sentences, target_sentences):
    src_input_ids, src_attention_masks = self.tokenize_strings(source_sentences)
    tgt_input_ids, tgt_attention_masks = self.tokenize_strings(target_sentences)
  
    batch = {
        "src_input_ids": src_input_ids,
        "src_attention_mask": src_attention_masks,
        "tgt_input_ids": tgt_input_ids,
        "tgt_attention_mask": tgt_attention_masks,
    }

    return batch

  def load_averitec_file(self, filepath):
    with open(filepath) as f:
      j = json.load(f)
      examples = j

    out_src = []
    out_tgt = []
    for example in examples:
        claim_str = self.extract_claim_str(example)

        claim_str.strip()
        out_src.append(claim_str)
        out_tgt.append(example["justification"].strip())

    print(len(out_src))
    print("===")
    
    return {"source": out_src, "target": out_tgt} 

  def extract_claim_str(self, example):
      claim_str = "[CLAIM] " + example["claim"] + " [EVIDENCE] "
      for question in example["questions"]:
          q_text = question["question"].strip()

          if len(q_text) == 0:
              continue

          if not q_text[-1] == "?":
              q_text += "?"

          answer_strings = []

          for answer in question["answers"]:
              if "answer_type" in answer and answer["answer_type"] == "Boolean":
                  answer_strings.append(answer["answer"] + ". " + answer["boolean_explanation"])
              else:
                  answer_strings.append(answer["answer"])

          claim_str += q_text
          for a_text in answer_strings:
              if not a_text[-1] == ".":
                  a_text += "."

              claim_str += " " + a_text.strip()

          claim_str += " "
      
      claim_str += " [VERDICT] " + example["label"]
      return claim_str    
 
  def prepare_data(self): # TODO set up to load answer only baseline
    self.train = self.load_averitec_file("/rds/user/mss84/hpc-work/datasets/averitec/full_data/date-cleaned.train.json")
    self.validate = self.load_averitec_file("/rds/user/mss84/hpc-work/datasets/averitec/full_data/date-cleaned.dev.json")
    self.test = self.load_averitec_file("/rds/user/mss84/hpc-work/datasets/averitec/full_data/date-cleaned.test.json")

  # encode the sentences using the tokenizer  
  def setup(self, stage):
    self.train = self.encode_sentences(self.train['source'], self.train['target'])
    self.validate = self.encode_sentences(self.validate['source'], self.validate['target'])
    self.test = self.encode_sentences(self.test['source'], self.test['target'])

  # Load the training, validation and test sets in Pytorch Dataset objects
  def train_dataloader(self):
    dataset = TensorDataset(self.train['src_input_ids'], self.train['src_attention_mask'], self.train['tgt_input_ids'], self.train['tgt_attention_mask'])                          
    train_data = DataLoader(dataset, sampler = RandomSampler(dataset), batch_size = self.batch_size)
    return train_data

  def val_dataloader(self):
    dataset = TensorDataset(self.validate['src_input_ids'], self.validate['src_attention_mask'], self.validate['tgt_input_ids'], self.validate['tgt_attention_mask']) 
    val_data = DataLoader(dataset, batch_size = self.batch_size)                       
    return val_data

  def test_dataloader(self):
    dataset = TensorDataset(self.test['src_input_ids'], self.test['src_attention_mask'], self.test['tgt_input_ids'], self.test['tgt_attention_mask']) 
    test_data = DataLoader(dataset, batch_size = self.batch_size)                   
    return test_data


