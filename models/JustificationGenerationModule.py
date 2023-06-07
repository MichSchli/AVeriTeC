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

def freeze_params(model):
  for layer in model.parameters():
    layer.requires_grade = False

class JustificationGenerationModule(pl.LightningModule):
    
  def __init__(self, tokenizer, model, learning_rate=1e-3, gen_num_beams=2, gen_max_length=100, should_pad_gen=True):
    super().__init__()
    self.tokenizer = tokenizer
    self.model = model
    self.learning_rate = learning_rate

    self.gen_num_beams = gen_num_beams
    self.gen_max_length = gen_max_length
    self.should_pad_gen = should_pad_gen

    #self.metrics =  datasets.load_metric('meteor')

    freeze_params(self.model.get_encoder())
    self.freeze_embeds()
  
  def freeze_embeds(self):
    ''' freeze the positional embedding parameters of the model; adapted from finetune.py '''
    freeze_params(self.model.model.shared)
    for d in [self.model.model.encoder, self.model.model.decoder]:
      freeze_params(d.embed_positions)
      freeze_params(d.embed_tokens)

  # Do a forward pass through the model
  def forward(self, input_ids, **kwargs):
    return self.model(input_ids, **kwargs)
  
  def configure_optimizers(self):
    optimizer = AdamW(self.parameters(), lr = self.learning_rate)
    return optimizer

  def shift_tokens_right(self, input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    https://github.com/huggingface/transformers/blob/main/src/transformers/models/bart/modeling_bart.py.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    if pad_token_id is None:
        raise ValueError("self.model.config.pad_token_id has to be defined.")
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids

  def run_model(self, batch):
    src_ids, src_mask, tgt_ids = batch[0], batch[1], batch[2]

    decoder_input_ids = self.shift_tokens_right(
                tgt_ids, self.tokenizer.pad_token_id, self.tokenizer.pad_token_id # BART uses the EOS token to start generation as well. Might have to change for other models.
            )

    outputs = self(src_ids, attention_mask=src_mask, decoder_input_ids=decoder_input_ids, use_cache=False)
    return outputs

  def compute_loss(self, batch):
    tgt_ids = batch[2]
    logits = self.run_model(batch)[0]

    cross_entropy = torch.nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
    loss = cross_entropy(logits.view(-1, logits.shape[-1]), tgt_ids.view(-1))

    return loss

  def training_step(self, batch, batch_idx):
    loss = self.compute_loss(batch)

    self.log("train_loss", loss, on_epoch=True)

    return {'loss':loss}

  def validation_step(self, batch, batch_idx):
    preds, loss, tgts = self.generate_and_compute_loss_and_tgts(batch)
    if self.should_pad_gen:
      preds = F.pad(preds, pad=(0, self.gen_max_length - preds.shape[1]), value=self.tokenizer.pad_token_id)

    self.log('val_loss', loss, prog_bar=True, sync_dist=True)

    return {'loss': loss, 'pred': preds, 'target': tgts}

  def test_step(self, batch, batch_idx):
    test_preds, test_loss, test_tgts = self.generate_and_compute_loss_and_tgts(batch)
    if self.should_pad_gen:
      test_preds = F.pad(test_preds, pad=(0, self.gen_max_length - test_preds.shape[1]), value=self.tokenizer.pad_token_id)

    self.log('test_loss', test_loss, prog_bar=True, sync_dist=True)

    return {'loss': test_loss, 'pred': test_preds, 'target': test_tgts}

  def test_epoch_end(self, outputs):
    self.handle_end_of_epoch_scoring(outputs, "test")

  def validation_epoch_end(self, outputs):
    self.handle_end_of_epoch_scoring(outputs, "val")

  def handle_end_of_epoch_scoring(self, outputs, prefix):
      gen = {}
      tgt = {}
      rouge = ROUGEScore()
      rouge_metric = lambda x, y: rouge(x,y)["rougeL_precision"]
      for out in outputs:
        preds = out['pred']
        tgts = out['target']

        preds = self.do_batch_detokenize(preds)
        tgts = self.do_batch_detokenize(tgts)

        for pred, t in zip(preds, tgts):
          rouge_d = rouge_metric(pred, t)
          self.log(prefix+"_rouge", rouge_d)

          meteor_d = pairwise_meteor(pred, t)
          self.log(prefix+"_meteor", meteor_d)

  def generate_and_compute_loss_and_tgts(self, batch):
    src_ids = batch[0]
    loss = self.compute_loss(batch)
    pred_ids, _ = self.generate_for_batch(src_ids)

    tgt_ids = batch[2]

    return pred_ids, loss, tgt_ids

  def do_batch_detokenize(self, batch):
    tokens = self.tokenizer.batch_decode(
      batch, 
      skip_special_tokens=True, 
      clean_up_tokenization_spaces=True
      )

    # Huggingface skipping of special tokens doesn't work for all models, so we do it manually as well for safety:
    tokens = [p.replace("<pad>", "") for p in tokens]
    tokens = [p.replace("<s>", "") for p in tokens]
    tokens = [p.replace("</s>", "") for p in tokens]

    return [t for t in tokens if t != ""]
  
  def generate_for_batch(self, batch):
    generated_ids = self.model.generate(
      batch, 
      decoder_start_token_id = self.tokenizer.pad_token_id,
      num_beams = self.gen_num_beams,
      max_length = self.gen_max_length
      )

    generated_tokens = self.tokenizer.batch_decode(
      generated_ids, 
      skip_special_tokens=True, 
      clean_up_tokenization_spaces=True
      )

    return generated_ids, generated_tokens


  def generate(self, text, max_input_length=512, device=None):
    encoded_dict = self.tokenizer(
            [text],
            max_length=max_input_length,
            padding="longest",
            truncation=True,
            return_tensors="pt",
            add_prefix_space = True
        )

    input_ids = encoded_dict['input_ids']

    if device is not None:
      input_ids = input_ids.to(device)

    with torch.no_grad():
        _, generated_tokens = self.generate_for_batch(input_ids)
    
    return generated_tokens[0]