import json
import pandas as pd
import torch
from bert_score import score

#select GPU
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

print(f"Using GPU is CUDA:{os.environ['CUDA_VISIBLE_DEVICES']}")

for i in range(torch.cuda.device_count()):
    info = torch.cuda.get_device_properties(i)
    print(f"CUDA:{i} {info.name}, {info.total_memory / 1024 ** 2}MB")

from transformers import DebertaForSequenceClassification, DebertaTokenizer, Trainer, TrainingArguments

from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import mlflow
import os

path_train = "SHROOM_unlabeled-training-data-v2/train.model-agnostic.json"
path_val = "SHROOM_dev-v2/val.model-agnostic.json"


def calculate_bertscore_tgt_hyp(df):
    '''a function that calculates bertscore between tgts and hyps'''
    tgts = df.tgt.to_list()
    hyps = df.hyp.to_list()

    P, R, F1 = score(hyps, tgts, lang='en', rescale_with_baseline=True)

    bertscore_values = list(map(float, F1))

    series_bertscore = pd.Series(bertscore_values)
    df['bertscore'] = series_bertscore.values

    return df


def calculate_bertscore_src_hyp(df):
    '''a function that calculates bertscore between src and hyps'''
    src = df.src.to_list()
    hyps = df.hyp.to_list()

    P, R, F1 = score(hyps, src, lang='en', rescale_with_baseline=True)

    bertscore_values = list(map(float, F1))

    series_bertscore = pd.Series(bertscore_values)
    df['bertscore'] = series_bertscore.values

    return df


with open(path_train) as f:
    train_data = json.load(f)

with open(path_val) as f:
    val_data = json.load(f)


#json to dataframe
df_train = pd.DataFrame.from_records(train_data)
df_val = pd.DataFrame.from_records(val_data)

#split train dataset into 2 subsets as PG part is missing tgt for bertscore
df_train_without_pg = df_train.loc[(df_train['task'] == 'MT') | (df_train['task'] == 'DM')].reset_index(drop=True)

df_train_pg = df_train.loc[(df_train['task'] == 'PG')].reset_index(drop=True)

calculate_bertscore_tgt_hyp(df_train_without_pg)

calculate_bertscore_src_hyp(df_train_pg)

#aggregate parts of the training set back
frames = [df_train_without_pg, df_train_pg]
df_train_with_bertscore = pd.concat(frames).reset_index(drop=True)
df_train_with_bertscore['hal_label_bertscore'] = df_train_with_bertscore["bertscore"] < 0.4 #set threshold to 0.4 for hallucination label
df_train_with_bertscore['hal_label_bertscore'] = df_train_with_bertscore['hal_label_bertscore'].map({False: 0, True: 1}) # 0: non-hallucination, 1: hallucination

#transform labels into numeric
df_val['label'] = df_val['label'].map({'Not Hallucination': 0, 'Hallucination': 1})


# DeBERTa tokenizer and model
tokenizer = DebertaTokenizer.from_pretrained('microsoft/deberta-base')
model = DebertaForSequenceClassification.from_pretrained('microsoft/deberta-base', num_labels=2)

# Tokenize and encode the dataset
train_encodings = tokenizer(list(df_train_with_bertscore['src']), list(df_train_with_bertscore['hyp']), truncation=True, padding=True)
valid_encodings = tokenizer(list(df_val['src']), list(df_val['hyp']), truncation=True, padding=True)

class SentencePairDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = SentencePairDataset(train_encodings, list(df_train_with_bertscore['hal_label_bertscore']))
valid_dataset = SentencePairDataset(valid_encodings, list(df_val['label']))

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='micro')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# Define TrainingArguments and Trainer
training_args = TrainingArguments(
    output_dir='./sentence_pair_classification_model_20240124',
    num_train_epochs=5,
    #per_device_train_batch_size=8,
    #per_device_eval_batch_size=8,
    #warmup_steps=500,
    #weight_decay=0.01,
    learning_rate=5e-5,
    evaluation_strategy = "epoch",
    save_strategy = "epoch",
    #logging_dir='./logs',
    #logging_steps=100,
    load_best_model_at_end=True,
)


os.environ["MLFLOW_EXPERIMENT_NAME"] = "shroom-bertscore-mlflow"
os.environ["MLFLOW_FLATTEN_PARAMS"] = "1"
#os.environ["MLFLOW_TRACKING_URI"]=""
os.environ["HF_MLFLOW_LOG_ARTIFACTS"]="1" 

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    compute_metrics=compute_metrics,
)

# Fine-tune the model
trainer.train()

mlflow.end_run()

# Save the model and tokenizer
model.save_pretrained('sentence_pair_classification_model')
tokenizer.save_pretrained('sentence_pair_classification_model')