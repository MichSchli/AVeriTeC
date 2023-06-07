import random
from torch.utils.data import DataLoader, TensorDataset, random_split, RandomSampler, Dataset
import pandas as pd
import numpy as np
import torch
import pytorch_lightning as pl
import json
from nltk.tokenize import word_tokenize
import tqdm

class SequenceClassificationDataLoader(pl.LightningDataModule):
  def __init__(self, tokenizer, data_file, batch_size, add_extra_nee=False):
    super().__init__()
    self.tokenizer = tokenizer
    self.data_file = data_file
    self.batch_size = batch_size
    self.add_extra_nee = add_extra_nee

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

  def encode_sentences(self, source_sentences, labels):
    input_ids = []
    attention_masks = []

    input_ids, attention_masks = self.tokenize_strings(source_sentences)

    labels = torch.as_tensor(labels)
  
    batch = {
        "input_ids": input_ids,
        "attention_mask": attention_masks,
        "labels": labels,
    }

    return batch

  def load_tsv_files(self, filepaths):
    srcs = []
    tgts = []

    for filepath in filepaths:
        with open(filepath) as tsv:
            for line in tsv:
                parts = line.strip().split("\t")

                if len(parts) == 2:
                    srcs.append(parts[0].strip())
                    tgts.append(int(parts[1].strip()))

    print(len(tgts))
    print("===")
    
    return {"source": srcs, "target": tgts}

  def quadruple_to_string(self, claim, question, answer, bool_explanation=""):
    if bool_explanation is not None and len(bool_explanation) > 0:
      bool_explanation = ", because " + bool_explanation.lower().strip()
    else:
      bool_explanation = ""
    return "[CLAIM] " + claim.strip() + " [QUESTION] " + question.strip() + " " + answer.strip() + bool_explanation

  def load_tsv_file(self, filepath):
    examples = self.load_tsv_files([filepath])

    return examples

  def load_averitec_file(self, filepath, add_extra_nee=False, qa_level_labels=False):
    label_map = {
      "Supported": 0,
      "Refuted": 1,
      "Not Enough Evidence": 2,
      "Conflicting Evidence/Cherrypicking": 3
    }

    with open(filepath) as f:
      j = json.load(f)
      examples = j

    data_points = []
    for example in examples:
      label = label_map[example["label"]]
      for question in example["questions"]:
        for answer in question["answers"]:
          if "boolean_explanation" in answer:
            quadruple = example["claim"], question["question"], answer["answer"], answer["boolean_explanation"]
          else:
            quadruple = example["claim"], question["question"], answer["answer"], ""

          if label == 3 and qa_level_labels: # Discard all conflicting evidence during training
            continue

          if answer["answer_type"] == "Unanswerable" and qa_level_labels: # Set unanswerable questions as nee during training
            this_dp_label = 2
          else:
            this_dp_label = label

          data_points.append((quadruple, this_dp_label))

    out = []
    for data_point, label in data_points:
      out.append((self.quadruple_to_string(*data_point), label))

    if add_extra_nee:
      for data_point, label in data_points:
        random_other_dp, _ = random.choice(data_points)
        while random_other_dp == data_point: # Reject if we select the same datapoint twice
          random_other_dp, _ = random.choice(data_points)
        
        out.append((self.quadruple_to_string(data_point[0], random_other_dp[1], random_other_dp[2], random_other_dp[3]), 2))

    print(len(out))
    print("===")
    
    return {"source": [o[0] for o in out], "target": [o[1] for o in out]}     
 
  def prepare_data(self): # TODO set up to load answer only baseline
    self.train = self.load_averitec_file(
      "/rds/user/mss84/hpc-work/datasets/averitec/full_data/date-cleaned.train.json", 
      add_extra_nee=self.add_extra_nee, 
      qa_level_labels=True)
    self.validate = self.load_averitec_file("/rds/user/mss84/hpc-work/datasets/averitec/full_data/date-cleaned.dev.json")
    self.test = self.load_averitec_file("/rds/user/mss84/hpc-work/datasets/averitec/full_data/date-cleaned.test.json")

  # encode the sentences using the tokenizer  
  def setup(self, stage):
    self.train = self.encode_sentences(self.train['source'], self.train['target'])
    
    self.validate = self.encode_sentences(self.validate['source'], self.validate['target'])
    
    self.test = self.encode_sentences(self.test['source'], self.test['target'])

  # Load the training, validation and test sets in Pytorch Dataset objects
  def train_dataloader(self):
    dataset = TensorDataset(self.train['input_ids'], self.train['attention_mask'], self.train['labels'])                          
    train_data = DataLoader(dataset, sampler = RandomSampler(dataset), batch_size = self.batch_size)
    return train_data

  def val_dataloader(self):
    dataset = TensorDataset(self.validate['input_ids'], self.validate['attention_mask'], self.validate['labels']) 
    val_data = DataLoader(dataset, batch_size = self.batch_size)                       
    return val_data

  def test_dataloader(self):
    dataset = TensorDataset(self.test['input_ids'], self.test['attention_mask'], self.test['labels']) 
    test_data = DataLoader(dataset, batch_size = self.batch_size)                   
    return test_data


