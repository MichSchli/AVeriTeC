import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from transformers import BertTokenizer, BertForSequenceClassification
import os
import sys
import argparse
import json
import tqdm
import torch

sys.path.insert(1, os.path.join(sys.path[0], '..'))

from data_loaders.NoEvidenceDataLoader import NoEvidenceDataLoader

from models.NaiveSeqClassModule import NaiveSeqClassModule

sys.path.insert(1, os.path.join(sys.path[0], '..'))

from data_loaders.SequenceClassificationDataLoader import SequenceClassificationDataLoader
from models.SequenceClassificationModule import SequenceClassificationModule
from utils import compute_all_pairwise_edit_distances, compute_all_pairwise_meteor_scores, delete_if_exists, edit_distance_custom, edit_distance_dbscan
from datasets import ClassLabel

parser = argparse.ArgumentParser(description='Perform veracity prediction using a sequence classifier without any evidence')
parser.add_argument('--averitec_file', default="data/dev.json", help='')
parser.add_argument('--train', action='store_true', help='Marks that training should happen. Otherwise, only inference is executed.')
parser.add_argument('--gpus', default=1, help='The number of available GPUs')
args = parser.parse_args()

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=4, problem_type="single_label_classification") # Must specify single_label for some reason

model = NaiveSeqClassModule(tokenizer = tokenizer, model = bert_model, learning_rate = 1e-5)

dataLoader = NoEvidenceDataLoader(
  tokenizer = tokenizer, 
  data_file = "this_is_discontinued", 
  batch_size = 32,
  add_extra_nee=True
  )

experiment_name = "bert_veracity-no_evidence"

checkpoint = ModelCheckpoint(
  dirpath='/rds/user/mss84/hpc-work/checkpoint_files/averitec',
  filename=experiment_name+"-{epoch:02d}-{val_loss:.2f}-{val_f1_epoch:.2f}", 
  save_top_k=1, 
  monitor="val_f1_epoch",
  mode="max"
  )

logger = pl.loggers.TensorBoardLogger(
                save_dir=os.getcwd(),
                version=experiment_name,
                name='lightning_logs'
            )

trainer = pl.Trainer(gpus=args.gpus,
  max_epochs=20,
  min_epochs=20,
  auto_lr_find=False,
  progress_bar_refresh_rate=1,
  callbacks=[checkpoint],
  logger=logger,
  accumulate_grad_batches=4,
  strategy="dp", #I tried ddp, it breaks
  num_nodes=1
)

if args.train:
    trainer.validate(model, dataLoader) # This makes pytorch lightning log initial values for dev loss etc. Nice for tensorboard.
    trainer.fit(model, dataLoader)
    best_checkpoint = checkpoint.best_model_path
    print("Finished training. The best checkpoint is stored at '" + best_checkpoint + "'.")
else:
    best_checkpoint = "/rds/user/mss84/hpc-work/checkpoint_files/averitec/bert_averitec_naive_extra_nee-epoch=16-val_loss=0.00-val_acc=0.00.ckpt"

trained_model = SequenceClassificationModule.load_from_checkpoint(best_checkpoint, tokenizer = tokenizer, model = bert_model)


if args.train:
    print("Running inference...")
    trainer.test(trained_model, dataLoader)
else:
    with open(args.averitec_file) as f:
        examples = json.load(f)

    for example in tqdm.tqdm(examples):
        example_strings = [example["claim"]]

        tokenized_strings, attention_mask = dataLoader.tokenize_strings(example_strings)
        example_support = torch.argmax(trained_model(tokenized_strings, attention_mask=attention_mask).logits, axis=1)
        
        label_map = {
          0: "Supported",
          1: "Refuted",
          2: "Not Enough Evidence",
          3: "Conflicting Evidence/Cherrypicking"
          }
        example["label"] = label_map[int(example_support)]

    print(json.dumps(examples, indent=4))