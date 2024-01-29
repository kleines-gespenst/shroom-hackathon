import json
import pandas as pd
import torch
import csv
from bert_score import score
from sentence_transformers import CrossEncoder, InputExample
from sentence_transformers.cross_encoder.evaluation import CEBinaryClassificationEvaluator
from sentence_transformers.evaluation import BinaryClassificationEvaluator
from torch.utils.data import DataLoader
import math
import mlflow
from sklearn.metrics import average_precision_score
import logging

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

logger = logging.getLogger(__name__)

threshold=0.47 #calculated based on AUC score

class ExtendedCEBinaryClassificationEvaluator(CEBinaryClassificationEvaluator):
    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1) -> dict:
        # Call the __call__ method from the parent class to get the initial metrics
        super().__call__(model, output_path=output_path, epoch=epoch, steps=steps)

        ap = super().__call__(model, output_path=output_path, epoch=epoch, steps=steps)

        # calculate other metrics based on the values returned by the parent class
        pred_scores = model.predict(
            self.sentence_pairs, convert_to_numpy=True, show_progress_bar=self.show_progress_bar
        )

        acc, acc_threshold = BinaryClassificationEvaluator.find_best_acc_and_threshold(pred_scores, self.labels, True)
        f1, precision, recall, f1_threshold = BinaryClassificationEvaluator.find_best_f1_and_threshold(
            pred_scores, self.labels, True
        )

        # Aggregate the metrics into a dictionary
        metrics = {
            "Accuracy": acc,
            "Average_Precision": ap,
            "F1": f1,
            "Precision": precision,
            "Recall": recall,
            }


        return acc, ap, f1, precision, recall
    

class ModifiedCrossEncoder(CrossEncoder):
    def _eval_during_training(self, evaluator, output_path, save_best_model, epoch, steps, callback):
        """Runs evaluation during training"""
        if evaluator is not None:
            acc, ap, f1, precision, recall = evaluator(self, output_path=output_path, epoch=epoch, steps=steps)
            if callback is not None:
                callback(ap, epoch, steps)
            if ap > self.best_score:
                self.best_score = ap
                if save_best_model:
                    self.save(output_path)


path_train_agnostic = "SHROOM_unlabeled-training-data-v2/train.model-agnostic.json"
path_train_aware = 'data_gpt_4_labeled/labeled-train.model-aware.v2.json'
val_agnostic_path = 'SHROOM_dev-v2/val.model-agnostic.json'
val_aware_path = 'SHROOM_dev-v2/val.model-aware.v2.json'

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


with open(path_train_agnostic) as f:
    train_agnostic = json.load(f)

with open(path_train_aware) as f:
    train_aware = json.load(f)

with open(val_agnostic_path) as f:
    val_agnostic = json.load(f)

with open(val_aware_path) as f:
    val_aware = json.load(f)


#json to dataframe
df_val_agnostic = pd.DataFrame.from_records(val_agnostic)
df_val_aware = pd.DataFrame.from_records(val_aware)
df_train_agnostic = pd.DataFrame.from_records(train_agnostic)
df_train_aware = pd.DataFrame.from_records(train_aware)
df_train_aware = df_train_aware.dropna() #there is one row with nan entry in src

#combine val sets for model-aware and model-agnostic tracks
frames_val = [df_val_agnostic, df_val_aware]
df_val = pd.concat(frames_val).reset_index(drop=True)

#split train dataset into 2 subsets as PG part is missing tgt for bertscore
df_train_agnostic_without_pg = df_train_agnostic.loc[(df_train_agnostic['task'] == 'MT') | (df_train_agnostic['task'] == 'DM')].reset_index(drop=True)

df_train_agnostic_pg = df_train_agnostic.loc[(df_train_agnostic['task'] == 'PG')].reset_index(drop=True)

df_train_aware_without_pg = df_train_aware.loc[(df_train_aware['task'] == 'MT') | (df_train_aware['task'] == 'DM')].reset_index(drop=True)

df_train_aware_pg = df_train_aware.loc[(df_train_aware['task'] == 'PG')].reset_index(drop=True)

calculate_bertscore_tgt_hyp(df_train_agnostic_without_pg)
calculate_bertscore_tgt_hyp(df_train_aware_without_pg)

calculate_bertscore_src_hyp(df_train_agnostic_pg)
calculate_bertscore_src_hyp(df_train_aware_pg)

#aggregate parts of the training set back
frames = [df_train_agnostic_without_pg, df_train_aware_without_pg, df_train_agnostic_pg, df_train_aware_pg]
df_train_with_bertscore = pd.concat(frames).reset_index(drop=True)
df_train_with_bertscore['hal_label_bertscore'] = df_train_with_bertscore["bertscore"] < threshold #set threshold based on AUC score
#df_train_with_bertscore['hal_label_bertscore'] = df_train_with_bertscore['hal_label_bertscore'].map({False: 0, True: 1}) # 0: non-hallucination, 1: hallucination

#transform labels into numeric
df_val['label'] = df_val['label'].map({'Not Hallucination': 0, 'Hallucination': 1})


# Number of epochs and other training configurations
num_epochs = 5
train_batch_size = 16  # Adjust based on your hardware capabilities
model_save_path = 'models/vectara_finetuned_bertscore'
model_name = 'vectara/hallucination_evaluation_model'
num_labels = 1


# Cache the dataset in memory to avoid reading from disk repeatedly
train_examples = [InputExample(texts=[entry['tgt'], entry['hyp']], label=float(1 - entry['bertscore'])) for entry in df_train_with_bertscore.to_dict('records')  if (entry['task'] == 'DM') or (entry['task'] == 'MT')]
#  1 src entry in the training set is None, so add if entry['src'] is not None else '' otherwise AttributeError: 'NoneType' object has no attribute 'strip'
train_examples += [InputExample(texts=[entry['src'] if entry['src'] is not None else '', entry['hyp']], label=float(1 - entry['bertscore'])) for entry in df_train_with_bertscore.to_dict('records')  if entry['task'] == 'PG']
# Convert to a list for efficient indexing
train_examples = list(train_examples)

#val examples
val_examples = [InputExample(texts=[entry['tgt'], entry['hyp']], label=int(1 - entry['label'])) for entry in df_val.to_dict('records') if (entry['task'] == 'DM') or (entry['task'] == 'MT')]
val_examples += [InputExample(texts=[entry['src'], entry['hyp']], label=int(1 - entry['label'])) for entry in df_val.to_dict('records') if entry['task'] == 'PG']
# Convert to a list for efficient indexing
val_examples = list(val_examples)

test_evaluator = ExtendedCEBinaryClassificationEvaluator.from_input_examples(val_examples, name='test_eval')

# Initialize the CrossEncoder model
model = ModifiedCrossEncoder(model_name, num_labels=num_labels, automodel_args={'ignore_mismatched_sizes': True})
# Set up the data loader for training
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=train_batch_size)
# Train the model

val_dataloader = DataLoader(val_examples, shuffle=True, batch_size=train_batch_size)
# Train the model
warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1)  # 10% of train data for warm-up


mlflow.set_experiment("/bertscore-vectara")


with mlflow.start_run():
    #mlflow.log_param('model_config', json.dumps(model.get_config_dict()))
    mlflow.log_param('num_epochs', num_epochs)
    mlflow.log_param('train_batch_size', train_batch_size)
    mlflow.log_param('model_save_path', model_save_path)
    mlflow.log_param('model_name', model_name)
    mlflow.log_param('num_labels', num_labels)

    # Initialize the LossLoggingCallback
    #loss_logging_callback = LossLoggingCallback()
    # callback=loss_logging_callback

    model.fit(train_dataloader=train_dataloader,
          evaluator=test_evaluator,
          epochs=num_epochs,
          evaluation_steps=10_000,
          warmup_steps=warmup_steps,
          output_path=model_save_path,
          show_progress_bar=True)

     # Evaluate on validation set after training
    evaluation_metrics = test_evaluator(model)
    print(evaluation_metrics)

    # Log all evaluation metrics
    metric_names = ["Accuracy", "Average_Precision", "F1", "Precision", "Recall"]
    for metric_name, metric_value in zip(metric_names, evaluation_metrics):
        mlflow.log_metric(f'eval_{metric_name}', metric_value)

        
    
    # Save the trained model explicitly
    model.save(model_save_path)

    # Log the model path as an artifact
    mlflow.log_artifact(model_save_path, artifact_path='models')

mlflow.end_run()



