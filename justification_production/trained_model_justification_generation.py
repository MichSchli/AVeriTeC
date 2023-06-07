import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from transformers import BartTokenizer, BartForConditionalGeneration, AdamW, BartConfig
import os
import sys
import argparse
import json
import tqdm
import torch

sys.path.insert(1, os.path.join(sys.path[0], '..'))

from data_loaders.JustificationProductionDataLoader import JustificationProductionDataLoader
from models.JustificationGenerationModule import JustificationGenerationModule

parser = argparse.ArgumentParser(description='Perform veracity prediction using a stance detection model..')
parser.add_argument('--averitec_file', default="data/dev.json", help='')
parser.add_argument('--train', action='store_true', help='Marks that training should happen. Otherwise, only inference is executed.')
parser.add_argument('--gpus', default=1, help='The number of available GPUs')
args = parser.parse_args()

tokenizer = BartTokenizer.from_pretrained('facebook/bart-large', add_prefix_space=True)
bart_model = BartForConditionalGeneration.from_pretrained("facebook/bart-large")
model = JustificationGenerationModule(tokenizer = tokenizer, model = bart_model, learning_rate = 1e-5)

dataLoader = JustificationProductionDataLoader(
  tokenizer = tokenizer, 
  batch_size = 32
  )

experiment_name = "bart_justifications_verdict"

checkpoint = ModelCheckpoint(
  dirpath='/rds/user/mss84/hpc-work/checkpoint_files/averitec',
  filename=experiment_name+"-{epoch:02d}-{val_loss:.2f}-{val_meteor:.2f}", 
  save_top_k=1, 
  monitor="val_meteor",
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
    best_checkpoint = "/rds/user/mss84/hpc-work/checkpoint_files/averitec/bart_justifications-epoch=16-val_loss=2.04-val_meteor=0.27.ckpt"

device = "cuda:0" if torch.cuda.is_available() else "cpu"
trained_model = JustificationGenerationModule.load_from_checkpoint(best_checkpoint, tokenizer = tokenizer, model = bart_model).to(device)

if args.train:
    print("Running inference...")
    trainer.test(trained_model, dataLoader)
else:
    with open(args.averitec_file) as f:
        examples = json.load(f)
    

    for example in tqdm.tqdm(examples):
        claim_str = dataLoader.extract_claim_str(example)
        claim_str.strip()

        example["justification"] = trained_model.generate(claim_str, device=device)

print(json.dumps(examples, indent=4))