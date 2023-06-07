import random
from torch.utils.data import DataLoader, TensorDataset, random_split, RandomSampler, Dataset
import pandas as pd
import numpy as np
import torch
import pytorch_lightning as pl
import json
from nltk.tokenize import word_tokenize
import tqdm

class NoEvidenceDataLoader(pl.LightningDataModule):
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
      claim = example["claim"]

      data_points.append((claim, label))

    print(len(data_points))
    print("===")
    
    return {"source": [o[0] for o in data_points], "target": [o[1] for o in data_points]}     
 
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


