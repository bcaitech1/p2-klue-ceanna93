import pickle as pickle
import os
import pandas as pd
import torch
from sklearn.metrics import accuracy_score
from transformers import AutoTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, BertConfig
from load_data import *
import numpy as np
import random

import argparse
from importlib import import_module
from pathlib import Path
import glob
import re

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
  MODEL_NAME = args.pretrained_model
  tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

  # load dataset
  train_dataset = load_data("/opt/ml/input/data/train/train.tsv")
  #dev_dataset = load_data("./dataset/train/dev.tsv")
  train_label = train_dataset['label'].values
  #dev_label = dev_dataset['label'].values
  
  # tokenizing dataset
  tokenized_train = bert_tokenized_dataset(train_dataset, tokenizer)
  #tokenized_dev = tokenized_dataset(dev_dataset, tokenizer)

  # make dataset for pytorch.
  RE_train_dataset = RE_Dataset(tokenized_train, train_label)
  #RE_dev_dataset = RE_Dataset(tokenized_dev, dev_label)

  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

  # setting model hyperparameter
  config_module = getattr(import_module("transformers"), args.model_type + "Config")
  model_config = config_module.from_pretrained(MODEL_NAME)
  model_config.num_labels = 42
  model_module = getattr(import_module("transformers"), args.model_type + "ForSequenceClassification")
  model = model_module.from_pretrained(MODEL_NAME, config=model_config)
  model.parameters
  model.to(device)

  output_dir = increment_output_dir(args.output_dir)
  
  # 사용한 option 외에도 다양한 option들이 있습니다.
  # https://huggingface.co/transformers/main_classes/trainer.html#trainingarguments 참고해주세요.
  training_args = TrainingArguments(
    output_dir=output_dir,          # output directory
    save_total_limit=args.save_total_limit,              # number of total save model.
    save_steps=args.save_steps,                 # model saving step.
    num_train_epochs=args.epochs,              # total number of training epochs
    learning_rate=args.lr,               # learning_rate
    per_device_train_batch_size=args.batch_size,  # batch size per device during training
    #per_device_eval_batch_size=16,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=100,              # log saving step.
  )
  trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=RE_train_dataset,         # training dataset
    #eval_dataset=RE_dev_dataset,             # evaluation dataset
    #compute_metrics=compute_metrics         # define metrics function
  )

  # train model
  trainer.train()

def main(args):
  train(args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=2020)
    parser.add_argument('--model_type', type=str, default='Bert')
    parser.add_argument('--pretrained_model', type=str, default='bert-base-multilingual-cased')
    
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--warmup_steps', type=int, default=500)               # number of warmup steps for learning rate scheduler
    parser.add_argument('--output_dir', type=str, default='./results/expr')
    parser.add_argument('--save_steps', type=int, default=100)
    parser.add_argument('--save_total_limit', type=int, default=3)
    parser.add_argument('--logging_steps', type=int, default=100)
    parser.add_argument('--logging_dir', type=str, default='./logs')            # directory for storing logs


    args = parser.parse_args()
    
    main(args)