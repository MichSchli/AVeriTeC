import pytorch_lightning as pl
import torch
import numpy as np
import datasets
from transformers import MaxLengthCriteria, StoppingCriteriaList
from transformers.optimization import AdamW
import itertools
from utils import count_stats, f1_metric, pairwise_meteor
from torchmetrics.text.rouge import ROUGEScore
import torch.nn.functional as F
import torchmetrics
from torchmetrics.classification import F1Score

class DualEncoderModule(pl.LightningModule):
  # Instantiate the model
  def __init__(self, tokenizer, model, learning_rate=1e-3):
    super().__init__()
    self.tokenizer = tokenizer
    self.model = model
    self.learning_rate = learning_rate

    self.train_acc = torchmetrics.Accuracy()
    self.val_acc = torchmetrics.Accuracy()
    self.test_acc = torchmetrics.Accuracy()

  # Do a forward pass through the model
  def forward(self, input_ids, **kwargs):
    return self.model(input_ids, **kwargs)
  
  def configure_optimizers(self):
    optimizer = AdamW(self.parameters(), lr = self.learning_rate)
    return optimizer

  def training_step(self, batch, batch_idx):
    pos_ids, pos_mask, neg_ids, neg_mask = batch

    neg_ids = neg_ids.view(-1, neg_ids.shape[-1])
    neg_mask = neg_mask.view(-1, neg_mask.shape[-1])

    pos_outputs = self(pos_ids, attention_mask=pos_mask, labels=torch.ones(pos_ids.shape[0], dtype=torch.uint8).to(pos_ids.get_device()))
    neg_outputs = self(neg_ids, attention_mask=neg_mask, labels=torch.zeros(neg_ids.shape[0], dtype=torch.uint8).to(neg_ids.get_device()))

    loss_scale = 1.0
    loss = pos_outputs.loss + loss_scale * neg_outputs.loss
    
    pos_logits = pos_outputs.logits
    pos_preds = torch.argmax(pos_logits, axis=1)
    self.train_acc(pos_preds.cpu(), torch.ones(pos_ids.shape[0], dtype=torch.uint8).cpu())      

    neg_logits = neg_outputs.logits
    neg_preds = torch.argmax(neg_logits, axis=1)
    self.train_acc(neg_preds.cpu(), torch.zeros(neg_ids.shape[0], dtype=torch.uint8).cpu())  


    return {'loss': loss}

  def validation_step(self, batch, batch_idx):
    pos_ids, pos_mask, neg_ids, neg_mask = batch

    neg_ids = neg_ids.view(-1, neg_ids.shape[-1])
    neg_mask = neg_mask.view(-1, neg_mask.shape[-1])

    pos_outputs = self(pos_ids, attention_mask=pos_mask, labels=torch.ones(pos_ids.shape[0], dtype=torch.uint8).to(pos_ids.get_device()))
    neg_outputs = self(neg_ids, attention_mask=neg_mask, labels=torch.zeros(neg_ids.shape[0], dtype=torch.uint8).to(neg_ids.get_device()))

    loss_scale = 1.0
    loss = pos_outputs.loss + loss_scale * neg_outputs.loss
    
    pos_logits = pos_outputs.logits
    pos_preds = torch.argmax(pos_logits, axis=1)
    self.val_acc(pos_preds.cpu(), torch.ones(pos_ids.shape[0], dtype=torch.uint8).cpu())      

    neg_logits = neg_outputs.logits
    neg_preds = torch.argmax(neg_logits, axis=1)
    self.val_acc(neg_preds.cpu(), torch.zeros(neg_ids.shape[0], dtype=torch.uint8).cpu())   
    
    self.log('val_acc', self.val_acc)    
    
    return {'loss': loss}

  def test_step(self, batch, batch_idx):
    pos_ids, pos_mask, neg_ids, neg_mask = batch

    neg_ids = neg_ids.view(-1, neg_ids.shape[-1])
    neg_mask = neg_mask.view(-1, neg_mask.shape[-1])

    pos_outputs = self(pos_ids, attention_mask=pos_mask, labels=torch.ones(pos_ids.shape[0], dtype=torch.uint8).to(pos_ids.get_device()))
    neg_outputs = self(neg_ids, attention_mask=neg_mask, labels=torch.zeros(neg_ids.shape[0], dtype=torch.uint8).to(neg_ids.get_device()))

    loss_scale = 1.0
    loss = pos_outputs.loss + loss_scale * neg_outputs.loss
    
    pos_logits = pos_outputs.logits
    pos_preds = torch.argmax(pos_logits, axis=1)
    self.test_acc(pos_preds.cpu(), torch.ones(pos_ids.shape[0], dtype=torch.uint8).cpu())      

    neg_logits = neg_outputs.logits
    neg_preds = torch.argmax(neg_logits, axis=1)
    self.test_acc(neg_preds.cpu(), torch.zeros(neg_ids.shape[0], dtype=torch.uint8).cpu())      


    self.log('test_acc', self.test_acc)
