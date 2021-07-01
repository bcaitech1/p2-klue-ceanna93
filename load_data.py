import pickle as pickle
import os
import pandas as pd
import torch
from pororo import Pororo
from tqdm import tqdm

ADDITIONAL_SPECIAL_TOKENS = ['BET']
ner = Pororo(task='ner', lang='ko')

# Dataset 구성.
class RE_Dataset(torch.utils.data.Dataset):
  def __init__(self, tokenized_dataset, labels):
    self.tokenized_dataset = tokenized_dataset
    self.labels = labels

  def __getitem__(self, idx):
    item = {key: torch.tensor(val[idx]) for key, val in self.tokenized_dataset.items()}
    item['labels'] = torch.tensor(self.labels[idx])
    return item

  def __len__(self):
    return len(self.labels)

# Dataset for ensemble
class ENSEMBLE_Dataset(torch.utils.data.Dataset):
  def __init__(self, xml_tokenized_dataset, hyper_tokenized_dataset, koelec_tokenized_dataset, labels):
    self.xml_tokenized_dataset = xml_tokenized_dataset
    self.hyper_tokenized_dataset = hyper_tokenized_dataset
    self.koelec_tokenized_dataset = koelec_tokenized_dataset
    self.labels = labels

  def __getitem__(self, idx):
    xitem = {key: torch.tensor(val[idx]) for key, val in self.xml_tokenized_dataset.items()}
    xitem['labels'] = torch.tensor(self.labels[idx])
    hitem = {key: torch.tensor(val[idx]) for key, val in self.hyper_tokenized_dataset.items()}
    hitem['labels'] = torch.tensor(self.labels[idx])
    kitem = {key: torch.tensor(val[idx]) for key, val in self.koelec_tokenized_dataset.items()}
    kitem['labels'] = torch.tensor(self.labels[idx])
    return xitem, hitem, kitem

  def __len__(self):
    return len(self.labels)

def add_entity(dataset):
    to_concat = pd.DataFrame(columns=['sentence','entity_01','entity_02','label'])

    antonym = {3:29, 4:10, 8:21, 15:23, 29:3, 10:4, 21:8, 23:15}
    same = [5, 6, 11, 24, 25]

    for i, row in dataset.iterrows():
        if(row['label'] in antonym.keys()):
            to_concat = to_concat.append({'sentence' : row['sentence'] , 'entity_01' : row['entity_02'], 'entity_02' : row['entity_01'], 'label' : antonym[row['label']]} , ignore_index=True)
        elif(row['label'] in same):
            to_concat = to_concat.append({'sentence' : row['sentence'] , 'entity_01' : row['entity_02'], 'entity_02' : row['entity_01'], 'label' : row['label']} , ignore_index=True)

    dataset=dataset.append(to_concat,ignore_index=True)
    return dataset

# 처음 불러온 tsv 파일을 원하는 형태의 DataFrame으로 변경 시켜줍니다.
# 변경한 DataFrame 형태는 baseline code description 이미지를 참고해주세요.
def preprocessing_dataset(dataset, label_type):
  label = []
  for i in dataset[8]:
    if i == 'blind':
      label.append(100)
    else:
      label.append(label_type[i])
  out_dataset = pd.DataFrame({'sentence':dataset[1],'entity_01':dataset[2],'entity_01_start':dataset[3],'entity_01_end':dataset[4],'entity_02':dataset[5],'entity_02_start':dataset[6],'entity_02_end':dataset[7],'label':label,})
#  out_dataset = pd.DataFrame({'sentence':dataset[1],'entity_01':dataset[2],'entity_02':dataset[5],'label':label,})
  
  if dataset[8][0] != 'blind':
    out_dataset.drop_duplicates(subset=['sentence', 'entity_01', 'entity_02'], inplace= True)
#    out_dataset = add_entity(out_dataset)

  return out_dataset

# tsv 파일을 불러옵니다.
def load_data(dataset_dir):
  # load label_type, classes
  with open('/opt/ml/input/data/label_type.pkl', 'rb') as f:
    label_type = pickle.load(f)
  # load dataset
  dataset = pd.read_csv(dataset_dir, delimiter='\t', header=None)
  # preprecessing dataset
  dataset = preprocessing_dataset(dataset, label_type)
  
  return dataset

# bert input을 위한 tokenizing.
# tip! 다양한 종류의 tokenizer와 special token들을 활용하는 것으로도 새로운 시도를 해볼 수 있습니다.
# baseline code에서는 2가지 부분을 활용했습니다.

def ko_tokenized_dataset(dataset, tokenizer):
  concat_entity = []
  for e01, e02 in zip(dataset['entity_01'], dataset['entity_02']):
    temp = ''
    temp = e01 + '[BET]' + e02
    concat_entity.append(temp)
  tokenized_sentences = tokenizer(
      concat_entity,
      list(dataset['sentence']),
      return_tensors="pt",
      padding=True,
      truncation=True,
      max_length=100,
      add_special_tokens=True,
      )
  tokenizer.add_special_tokens({'additional_special_tokens':ADDITIONAL_SPECIAL_TOKENS})
  return tokenized_sentences

def tokenized_dataset(dataset, tokenizer):
  concat_entity = []
  for e01, e02 in zip(dataset['entity_01'], dataset['entity_02']):
    temp = ''
    temp = e01 + '</s>' + e02
    concat_entity.append(temp)
  tokenized_sentences = tokenizer(
      concat_entity,
      list(dataset['sentence']),
      return_tensors="pt",
      padding=True,
      truncation=True,
      max_length=100,
      add_special_tokens=True,
      )
  return tokenized_sentences


def tem_tokenized_dataset(dataset, tokenizer):
  fixed_sent = []
  print('start processing...')
  for sent, ent01, ent02, start1, end1, start2, end2 in tqdm(zip(dataset['sentence'], dataset['entity_01'], dataset['entity_02'],dataset['entity_01_start'], dataset['entity_01_end'], dataset['entity_02_start'], dataset['entity_02_end'])):
    ner_01 = ' ※ ' + ner(ent01)[0][1].lower() + ' ※ '
    ner_02 = ' ∧ ' + ner(ent02)[0][1].lower() + ' ∧ '

    entity_01_start, entity_01_end = int(start1), int(end1)
    entity_02_start, entity_02_end = int(start2), int(end2)
    if entity_01_start < entity_02_start:
      sent = sent[:entity_01_start] + '§' + ner_01 + sent[entity_01_start:entity_01_end+1] + ' § ' + sent[entity_01_end+1:entity_02_start] + '¿' + ner_02 + sent[entity_02_start:entity_02_end+1] + ' ¿ ' + sent[entity_02_end+1:]
    else:
      sent = sent[:entity_02_start] + '¿' + ner_02 + sent[entity_02_start:entity_02_end+1] + ' [¿ ' + sent[entity_02_end+1:entity_01_start] + '§' + ner_01 + sent[entity_01_start:entity_01_end+1] + ' § ' + sent[entity_01_end+1:]
    fixed_sent.append(sent)
  concat_entity = []
  for e01, e02 in zip(dataset['entity_01'], dataset['entity_02']):
    temp = ''
    temp = e01 + '</s>' + e02
    concat_entity.append(temp)
  tokenized_sentences = tokenizer(
      concat_entity,
      list(fixed_sent),
      return_tensors="pt",
      padding=True,
      truncation=True,
      max_length=100,
      add_special_tokens=True,
      )
  #tokenizer.add_special_tokens({"additional_special_tokens": ADDITIONAL_SPECIAL_TOKENS})
  return tokenized_sentences