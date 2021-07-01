import pickle as pickle
import os
import pandas as pd
import torch
import numpy as np
import random
from sklearn.metrics import accuracy_score
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification, Trainer, TrainingArguments, XLMRobertaConfig, ElectraForSequenceClassification, ElectraTokenizer
from load_data import *

import argparse
from importlib import import_module
from pathlib import Path
import glob
import re

# seed 고정 
def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

# 평가를 위한 metrics function.
def compute_metrics(pred):
  labels = pred.label_ids
  preds = pred.predictions.argmax(-1)
  # calculate accuracy using sklearn's function
  acc = accuracy_score(labels, preds)
  return {
      'accuracy': acc,
  }

def increment_output_dir(output_path, exist_ok=False):
  path = Path(output_path)
  if (path.exists() and exist_ok) or (not path.exists()):
    return str(path)
  else:
    dirs = glob.glob(f"{path}*")
    matches = [re.search(rf"%s(\d+)" %path.stem, d) for d in dirs]
    i = [int(m.groups()[0]) for m in matches if m]
    n = max(i) + 1 if i else 2
    return f"{path}{n}"

def train(args):
  seed_everything(args.seed)
  # load model and tokenizer
#  MODEL_NAME = "xlm-roberta-large"
#  tokenizer = XLMRobertaTokenizer.from_pretrained(MODEL_NAME)
  MODEL_NAME = "monologg/koelectra-base-v3-discriminator"
  tokenizer = ElectraTokenizer.from_pretrained(MODEL_NAME)

  # load dataset
  train_dataset = load_data("/opt/ml/input/data/train/train.tsv")
  #dev_dataset = load_data("./dataset/train/train_dev.tsv")
  train_label = train_dataset['label'].values
  #dev_label = dev_dataset['label'].values

  # tokenizing dataset
  tokenized_train = ko_tokenized_dataset(train_dataset, tokenizer)
  #tokenized_dev = tokenized_dataset(dev_dataset, tokenizer)

  # make dataset for pytorch.
  RE_train_dataset = RE_Dataset(tokenized_train, train_label)
  #RE_dev_dataset = RE_Dataset(tokenized_dev, dev_label)

  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

  # setting model hyperparameter
#  bert_config = XLMRobertaConfig.from_pretrained(MODEL_NAME)
#  bert_config.num_labels = 42
#  model = XLMRobertaForSequenceClassification.from_pretrained(MODEL_NAME, config=bert_config)
#  model.resize_token_embeddings(len(tokenizer))
  config_module = getattr(import_module("transformers"), "ElectraConfig")
  model_config = config_module.from_pretrained(MODEL_NAME)
  model_config.num_labels = 42
  model = ElectraForSequenceClassification.from_pretrained(MODEL_NAME, config=model_config)
  model.resize_token_embeddings(len(tokenizer))
  model.parameters
  model.to(device)

  output_dir = increment_output_dir(args.output_dir)
  
  # 사용한 option 외에도 다양한 option들이 있습니다.
  # https://huggingface.co/transformers/main_classes/trainer.html#trainingarguments 참고해주세요.
  training_args = TrainingArguments(
    output_dir=output_dir,          # output directory
    save_total_limit=args.save_total_limit,              # number of total save model.
    num_train_epochs=args.epochs,              # total number of training epochs
    learning_rate=args.lr,               # learning_rate
    per_device_train_batch_size=args.batch_size,  # batch size per device during training
    warmup_steps=args.warmup_steps,                # number of warmup steps for learning rate scheduler
    weight_decay=args.weight_decay,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=100,              # log saving step.
    save_steps=100,
    dataloader_num_workers=4,
    label_smoothing_factor=args.label_smoothing_factor,
  )
  trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=RE_train_dataset,         # training dataset
    compute_metrics = compute_metrics,
  )

  # train model
  trainer.train()

def main(args):
  train(args)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--seed', type=int, default=142)
  parser.add_argument('--epochs', type=int, default=10)
  parser.add_argument('--batch_size', type=int, default=32)
  parser.add_argument('--lr', type=float, default=1e-5)
  parser.add_argument('--weight_decay', type=float, default=0.01)
  parser.add_argument('--warmup_steps', type=int, default=300)               # number of warmup steps for learning rate scheduler
  parser.add_argument('--output_dir', type=str, default='./results/expr')
  parser.add_argument('--save_steps', type=int, default=100)
  parser.add_argument('--save_total_limit', type=int, default=1)
  parser.add_argument('--logging_steps', type=int, default=100)
  parser.add_argument('--logging_dir', type=str, default='./logs')            # directory for storing logs
  parser.add_argument('--label_smoothing_factor', type=float, default=0.5)            # directory for storing logs


  args = parser.parse_args()
  
  main(args)