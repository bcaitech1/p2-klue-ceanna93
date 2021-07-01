from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification, Trainer, TrainingArguments, XLMRobertaConfig, ElectraTokenizer, ElectraForSequenceClassification
from torch.utils.data import DataLoader
from load_data import *
import pandas as pd
import torch
import pickle as pickle
import numpy as np
import argparse

# import argparse
from importlib import import_module

def inference(model, tokenized_sent, device):
  dataloader = DataLoader(tokenized_sent, batch_size=40, shuffle=False)
  model.eval()
  output_pred = []
  
  for i, data in enumerate(dataloader):
    with torch.no_grad():
      outputs = model(
          input_ids=data['input_ids'].to(device),
          attention_mask=data['attention_mask'].to(device),
        #  token_type_ids=data['token_type_ids'].to(device)
          )
    logits = outputs[0]
    logits = logits.detach().cpu().numpy()
    result = np.argmax(logits, axis=-1)

    output_pred.append(result)
  
  return np.array(output_pred).flatten()

# tokenzied_sent_set에는 각 모델 별 다른 dataset이 들어갈 예정
def ensemble_inference(xml_model, hyper_model, koelec_model, tokenzied_sent_set, device):
  dataloader = DataLoader(tokenzied_sent_set, batch_size=40, shuffle=False)
  xml_model.eval()
  hyper_model.eval()
  koelec_model.eval()
  output_pred = []
  
  print(tokenzied_sent_set)
  for i, data in enumerate(dataloader):
    with torch.no_grad():
      xml_pred = xml_model(
          input_ids=data[0]['input_ids'].to(device),
          attention_mask=data[0]['attention_mask'].to(device)
          )
      hyper_pred = hyper_model(
          input_ids=data[1]['input_ids'].to(device),
          attention_mask=data[1]['attention_mask'].to(device)
          )
      koelec_pred = koelec_model(
          input_ids=data[2]['input_ids'].to(device),
          attention_mask=data[2]['attention_mask'].to(device),
          token_type_ids=data[2]['token_type_ids'].to(device)
          )
    xml_logits = xml_pred[0].mul(0.77)
    hyper_logits = hyper_pred[0].mul(0.78)
    koelec_logits = koelec_pred[0].mul(0.74)
    st = torch.stack([xml_logits, hyper_logits, koelec_logits])
    pred = torch.mean(st, dim=0)

    pred = pred.argmax(dim=-1)

    output_pred.append(pred.cpu().numpy())
  
  return np.array(output_pred).flatten()

def load_test_dataset(dataset_dir, tokenizer):
  test_dataset = load_data(dataset_dir)
  test_label = test_dataset['label'].values
  # tokenizing dataset
  tokenized_test = ko_tokenized_dataset(test_dataset, tokenizer)
  return tokenized_test, test_label

def load_ensemble_test_dataset(dataset_dir, xml_tokenizer, hy_tokenizer, ko_tokenizer):
  test_dataset = load_data(dataset_dir)
  test_label = test_dataset['label'].values
  # tokenizing dataset
  xml_tokenized_test = tem_tokenized_dataset(test_dataset, xml_tokenizer)
  hyper_tokenized_test = tokenized_dataset(test_dataset, hy_tokenizer)
  koelec_tokenized_test = ko_tokenized_dataset(test_dataset, ko_tokenizer)
  return xml_tokenized_test, hyper_tokenized_test, koelec_tokenized_test, test_label

def main(args):
  """
    주어진 dataset tsv 파일과 같은 형태일 경우 inference 가능한 코드입니다.
  """
  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  print(args)
  if args.is_ensemble == True:
    xml_tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-large')
    hyper_tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-large')
    koelec_tokenizer = ElectraTokenizer.from_pretrained('monologg/koelectra-base-v3-discriminator')

    xml_model = XLMRobertaForSequenceClassification.from_pretrained(args.xml_model_dir).to(device)
    hyper_model = XLMRobertaForSequenceClassification.from_pretrained(args.hyper_model_dir).to(device)
    koelec_model = ElectraForSequenceClassification.from_pretrained(args.koelec_model_dir).to(device)

    test_dataset_dir = "/opt/ml/input/data/test/test.tsv"
    xml_dataset, hyper_dataset, koelec_dataset, test_label = load_ensemble_test_dataset(test_dataset_dir, xml_tokenizer, hyper_tokenizer, koelec_tokenizer)
    test_dataset = ENSEMBLE_Dataset(xml_dataset, hyper_dataset, koelec_dataset, test_label)

    pred_answer = ensemble_inference(xml_model, hyper_model, koelec_model, test_dataset, device)

    output = pd.DataFrame(pred_answer, columns=['pred'])

    if os.path.exists(args.out_path):
      output.to_csv(args.out_path + "/submission_ensemble.csv", index=False)

  else:
  # load tokenizer
#    MODEL_NAME = "xlm-roberta-large"
#    tokenizer = XLMRobertaTokenizer.from_pretrained(MODEL_NAME)
    MODEL_NAME = "monologg/koelectra-base-v3-discriminator"
    tokenizer = ElectraTokenizer.from_pretrained(MODEL_NAME)

    # load my model
#    model = XLMRobertaForSequenceClassification.from_pretrained(args.model_dir)
    model = ElectraForSequenceClassification.from_pretrained(args.model_dir)
    model.to(device)

    # load test datset
    test_dataset_dir = "/opt/ml/input/data/test/test.tsv"
    test_dataset, test_label = load_test_dataset(test_dataset_dir, tokenizer)
    test_dataset = RE_Dataset(test_dataset ,test_label)

    # predict answer
    pred_answer = inference(model, test_dataset, device)
    # make csv file with predicted answer
    # 아래 directory와 columns의 형태는 지켜주시기 바랍니다.

    output = pd.DataFrame(pred_answer, columns=['pred'])
  #   output.to_csv('./prediction/submission.csv', index=False)

    if os.path.exists(args.out_path):
      output.to_csv(args.out_path + "/submission.csv", index=False)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  
  # model dir
  parser.add_argument('--model_dir', type=str, default="./results/expr/checkpoint-2500")
  parser.add_argument('--out_path', type=str, default="./prediction")
  parser.add_argument('--is_ensemble', type=bool, default=False)
  parser.add_argument('--xml_model_dir', type=str, default="./results/expr/checkpoint-2500")
  parser.add_argument('--hyper_model_dir', type=str, default="./results/expr/checkpoint-2500")
  parser.add_argument('--koelec_model_dir', type=str, default="./results/expr/checkpoint-2500")
  args = parser.parse_args()
  
  main(args)
