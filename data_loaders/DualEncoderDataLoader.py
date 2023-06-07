from torch.utils.data import DataLoader, TensorDataset, random_split, RandomSampler, Dataset
import pandas as pd
import numpy as np
import torch
import pytorch_lightning as pl
import json
from nltk.tokenize import word_tokenize
import tqdm
import random

class DualEncoderDataLoader(pl.LightningDataModule):
  def __init__(self, tokenizer, data_file, batch_size):
    super().__init__()
    self.tokenizer = tokenizer
    self.data_file = data_file
    self.batch_size = batch_size

  def encode_sentences(self, positives, negatives, max_length=512, pad_to_max_length=False, return_tensors="pt"):
    encoded_dict = self.tokenizer(
            positives,
            max_length=max_length,
            padding="max_length" if pad_to_max_length else "longest",
            truncation=True,
            return_tensors=return_tensors
        )

    pos_input_ids = encoded_dict['input_ids']
    pos_attention_masks = encoded_dict['attention_mask']

    negatives_unstacked = list(np.reshape(negatives, -1))

    encoded_dict = self.tokenizer(
            negatives_unstacked,
            max_length=max_length,
            padding="max_length" if pad_to_max_length else "longest",
            truncation=True,
            return_tensors=return_tensors
        )

    neg_input_ids = encoded_dict['input_ids'].view(len(negatives), len(negatives[0]), -1)
    neg_attention_masks = encoded_dict['attention_mask'].view(len(negatives), len(negatives[0]), -1)
  
    batch = {
        "pos_input_ids": pos_input_ids,
        "pos_attention_masks": pos_attention_masks,
        "neg_input_ids": neg_input_ids,
        "neg_attention_masks": neg_attention_masks,
    }

    return batch

  def triple_to_string(self, x):
    return " </s> ".join([item.strip() for item in x])
  
  def load_tsv_files(self, filepaths):
    srcs = []

    for filepath in filepaths:
        with open(filepath) as tsv:
            for line in tsv:
                parts = line.strip().split("\t")

                if len(parts) == 2:
                    claim = parts[0].split("||")[0].replace("[CLAIM]", "").strip()
                    question = parts[0].split("||")[1].replace("[QUESTION]", "").strip()
                    answer = parts[1].split("||")[1]

                    srcs.append([claim, question, answer])
   
    return {"positives": srcs}

  def add_negatives(self, dataset, neg_count=3):
    srcs = dataset["positives"]
    negatives = []

    for i,src in enumerate(srcs):
        negatives.append([])
        for _ in range(neg_count):
            neg = random.randint(0, len(srcs)-1)
            while srcs[neg][0] == srcs[i][0] and srcs[neg][1] == srcs[neg][1]: # Reject if we happen to sample the same claim/question pair
                neg = random.randint(0, len(srcs)-1)

            claim_neg = (srcs[neg][0], src[1], src[2])
            question_neg = (src[0], srcs[neg][1], src[2])
            answer_neg = (src[0], src[1], srcs[neg][2])

            negatives[-1].append(self.triple_to_string(claim_neg))
            negatives[-1].append(self.triple_to_string(question_neg))
            negatives[-1].append(self.triple_to_string(answer_neg))

    return {"positives": [self.triple_to_string(s) for s in srcs], "negatives": negatives}


  def load_averitec_file(self, filepath):
    with open(filepath) as f:
      j = json.load(f)
      examples = j

    srcs = []
    for example in examples:
      for question in example["questions"]:
        for answer in question["answers"]:
          if "answer" not in answer:
            continue
          a = answer["answer"]
          if answer["answer_type"] == "Boolean":
            a += ". " + answer["boolean_explanation"]
          srcs.append([example["claim"], question["question"], a])

    return {"positives": srcs}

  def load_tsv_file(self, filepath):
    return self.load_tsv_files([filepath])

  def load_dataset_files(self, averitec_filepaths, tsv_filepaths):
    averitec_data = [self.load_averitec_file(f) for f in averitec_filepaths]
    tsv_data = self.load_tsv_files(tsv_filepaths)

    all_data = {"positives": []}

    for data in averitec_data:
      all_data["positives"].extend(data["positives"])

    all_data["positives"].extend(tsv_data["positives"])

    return all_data
 
  # Loads and splits the data into training, validation and test sets with a 60/20/20 split
  def prepare_data(self):
    #self.train = self.load_averitec_file("/rds/user/mss84/hpc-work/datasets/averitec/full_data/date-cleaned.train.json")
    self.train = self.load_dataset_files(
      ["/rds/user/mss84/hpc-work/datasets/averitec/full_data/date-cleaned.train.json"],
      ["/rds/user/mss84/hpc-work/datasets/averitec/initial_test_dataset/fcb.train_retriever.tsv"]
    )
    self.train = self.add_negatives(self.train)

    self.validate = self.load_averitec_file("/rds/user/mss84/hpc-work/datasets/averitec/full_data/date-cleaned.dev.json")
    self.validate = self.add_negatives(self.validate)

    self.test = self.load_averitec_file("/rds/user/mss84/hpc-work/datasets/averitec/full_data/date-cleaned.test.json")
    self.test = self.add_negatives(self.test)

  # encode the sentences using the tokenizer  
  def setup(self, stage):
    self.train = self.encode_sentences(self.train['positives'], self.train['negatives'])
    
    self.validate = self.encode_sentences(self.validate['positives'], self.validate['negatives'])
    
    self.test = self.encode_sentences(self.test['positives'], self.test['negatives'])

  # Load the training, validation and test sets in Pytorch Dataset objects
  def train_dataloader(self):
    dataset = TensorDataset(self.train['pos_input_ids'], self.train['pos_attention_masks'], self.train['neg_input_ids'], self.train['neg_attention_masks'])                          
    train_data = DataLoader(dataset, sampler = RandomSampler(dataset), batch_size = self.batch_size)
    return train_data

  def val_dataloader(self):
    dataset = TensorDataset(self.validate['pos_input_ids'], self.validate['pos_attention_masks'], self.validate['neg_input_ids'], self.validate['neg_attention_masks']) 
    val_data = DataLoader(dataset, batch_size = self.batch_size)                       
    return val_data

  def test_dataloader(self):
    dataset = TensorDataset(self.test['pos_input_ids'], self.test['pos_attention_masks'], self.test['neg_input_ids'], self.test['neg_attention_masks']) 
    test_data = DataLoader(dataset, batch_size = self.batch_size)                   
    return test_data


