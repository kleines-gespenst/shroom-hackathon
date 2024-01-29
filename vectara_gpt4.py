import json
import pandas as pd
from sentence_transformers import CrossEncoder, InputExample
from sentence_transformers.cross_encoder.evaluation import CEBinaryClassificationEvaluator
from torch.utils.data import DataLoader
import math
import mlflow
import torch
import mlflow.pytorch 
import csv
from sklearn.metrics import average_precision_score
import logging

#select GPU
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

print(f"Using GPU is CUDA:{os.environ['CUDA_VISIBLE_DEVICES']}")

for i in range(torch.cuda.device_count()):
    info = torch.cuda.get_device_properties(i)
    print(f"CUDA:{i} {info.name}, {info.total_memory / 1024 ** 2}MB")

# Set your MLflow experiment name
#mlflow_experiment_name = "vectara-gpt4-labeled"
#mlflow.set_experiment(mlflow_experiment_name)

# training model-aware dataset
file_path_training_aware = 'data_gpt_4_labeled/labeled-train.model-aware.v2.json'
with open(file_path_training_aware, 'r') as f:
    train_aware = json.load(f)

# Convert the model-aware training set to a DataFrame
df_train_aware = pd.DataFrame(train_aware)

# training model-agnostic dataset
file_path_training_agnostic = 'data_gpt_4_labeled/labeled-train.model-agnostic.json'
with open(file_path_training_agnostic, 'r') as f:
    train_agnostic = json.load(f)
# Convert the model-agnostic training set to a DataFrame
df_train_agnostic = pd.DataFrame(train_agnostic)

# Concatenate the existing and new DataFrames
df_train = pd.concat([df_train_aware, df_train_agnostic], ignore_index=True)

# Val set
val_agnostic_path = 'SHROOM_dev-v2/val.model-agnostic.json'
val_aware_path = 'SHROOM_dev-v2/val.model-aware.v2.json'

with open(val_agnostic_path) as f:
    val_agnostic = json.load(f)

with open(val_aware_path) as f:
    val_aware = json.load(f)

#json to dataframe
df_val_agnostic = pd.DataFrame.from_records(val_agnostic)
df_val_aware = pd.DataFrame.from_records(val_aware)

#combine val sets for model-aware and model-agnostic tracks
frames_val = [df_val_agnostic, df_val_aware]
df_val = pd.concat(frames_val).reset_index(drop=True)

# Number of epochs and other training configurations
num_epochs = 5
train_batch_size = 16  # Adjust based on your hardware capabilities
model_save_path = 'models/vectara_finetuned_gpt4labels_mlflow'
model_name = 'vectara/hallucination_evaluation_model'
num_labels = 1


# Cache the dataset in memory to avoid reading from disk repeatedly
train_examples = [InputExample(texts=[entry['tgt'], entry['hyp']], label=float(1 - entry['p(Hallucination)'])) for entry in df_train.to_dict('records')  if (entry['task'] == 'DM') or (entry['task'] == 'MT')]
#  1 src entry in the training set is None, so add if entry['src'] is not None else '' otherwise AttributeError: 'NoneType' object has no attribute 'strip'
train_examples += [InputExample(texts=[entry['src'] if entry['src'] is not None else '', entry['hyp']], label=float(1 - entry['p(Hallucination)'])) for entry in df_train.to_dict('records')  if entry['task'] == 'PG']
# Convert to a list for efficient indexing
train_examples = list(train_examples)

#val examples
val_examples = [InputExample(texts=[entry['tgt'], entry['hyp']], label=float(1 - entry['p(Hallucination)'])) for entry in df_val.to_dict('records') if (entry['task'] == 'DM') or (entry['task'] == 'MT')]
val_examples += [InputExample(texts=[entry['src'], entry['hyp']], label=float(1 - entry['p(Hallucination)'])) for entry in df_val.to_dict('records') if entry['task'] == 'PG']
# Convert to a list for efficient indexing
val_examples = list(val_examples)

test_evaluator = CEBinaryClassificationEvaluator.from_input_examples(val_examples, name='test_eval')

# Initialize the CrossEncoder model
model = CrossEncoder(model_name, num_labels=num_labels, automodel_args={'ignore_mismatched_sizes': True})
# Set up the data loader for training
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=train_batch_size)
# Train the model

val_dataloader = DataLoader(val_examples, shuffle=True, batch_size=train_batch_size)
# Train the model
warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1)  # 10% of train data for warm-up

logger = logging.getLogger(__name__)

class CEBinaryClassificationEvaluator:
    # ... (rest of the class code)

    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1) -> dict:
        result_dict = {}

        logger.info("CEBinaryClassificationEvaluator: Evaluating the model on " + self.name + " dataset.")
        pred_scores = model.predict(
            self.sentence_pairs, convert_to_numpy=True, show_progress_bar=self.show_progress_bar
        )

        acc, acc_threshold = CEBinaryClassificationEvaluator.find_best_acc_and_threshold(pred_scores, self.labels, True)
        f1, precision, recall, f1_threshold = CEBinaryClassificationEvaluator.find_best_f1_and_threshold(
            pred_scores, self.labels, True
        )
        ap = average_precision_score(self.labels, pred_scores)

        logger.info("Accuracy:           {:.2f}\t(Threshold: {:.4f})".format(acc * 100, acc_threshold))
        logger.info("F1:                 {:.2f}\t(Threshold: {:.4f})".format(f1 * 100, f1_threshold))
        logger.info("Precision:          {:.2f}".format(precision * 100))
        logger.info("Recall:             {:.2f}".format(recall * 100))
        logger.info("Average Precision:  {:.2f}".format(ap * 100))

        # Log metrics to result_dict
        result_dict['accuracy'] = acc
        result_dict['accuracy_threshold'] = acc_threshold
        result_dict['f1'] = f1
        result_dict['f1_threshold'] = f1_threshold
        result_dict['precision'] = precision
        result_dict['recall'] = recall
        result_dict['average_precision'] = ap

        if output_path is not None and self.write_csv:
            csv_path = os.path.join(output_path, self.csv_file)
            output_file_exists = os.path.isfile(csv_path)
            with open(csv_path, mode="a" if output_file_exists else "w", encoding="utf-8") as f:
                writer = csv.writer(f)
                if not output_file_exists:
                    writer.writerow(self.csv_headers)

                writer.writerow([epoch, steps, acc, acc_threshold, f1, f1_threshold, precision, recall, ap])

        return result_dict


mlflow.set_experiment("/mlflow-vectara-gpt4")


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
    evaluation_metrics = test_evaluator(model, val_dataloader)

    # Log all evaluation metrics
    for metric_name, metric_value in evaluation_metrics.items():
        mlflow.log_metric(f'eval_{metric_name}', metric_value)

        
    
    # Save the trained model explicitly
    model.save(model_save_path)

    # Log the trained model as an artifact
    mlflow.pytorch.log_model(model, artifact_path='models')

mlflow.end_run()



