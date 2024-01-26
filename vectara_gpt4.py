import json
import pandas as pd
from sentence_transformers import CrossEncoder, InputExample
from sentence_transformers.cross_encoder.evaluation import CEBinaryClassificationEvaluator
from torch.utils.data import DataLoader
import math
import mlflow
import torch

#select GPU
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

print(f"Using GPU is CUDA:{os.environ['CUDA_VISIBLE_DEVICES']}")

for i in range(torch.cuda.device_count()):
    info = torch.cuda.get_device_properties(i)
    print(f"CUDA:{i} {info.name}, {info.total_memory / 1024 ** 2}MB")

# Set your MLflow experiment name
mlflow_experiment_name = "vectara-gpt4-labeled"
mlflow.set_experiment(mlflow_experiment_name)

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
model_save_path = 'models/vectara_finetuned_gpt4labels'
model_name = 'vectara/hallucination_evaluation_model'
num_labels = 1

# Start MLflow run
with mlflow.start_run():

    # Log parameters
    mlflow.log_param("num_epochs", num_epochs)
    mlflow.log_param("train_batch_size", train_batch_size)
    mlflow.log_param("model_save_path", model_save_path)
    mlflow.log_param("model_name", model_name)
    mlflow.log_param("num_labels", num_labels)

    # Cache the dataset in memory to avoid reading from disk repeatedly
    train_examples = [InputExample(texts=[entry['tgt'], entry['hyp']], label=int(1 - entry['p(Hallucination)'])) for entry in df_train.to_dict('records')]
    # Convert to a list for efficient indexing
    train_examples = list(train_examples)

    # Initialize the CrossEncoder model
    model = CrossEncoder(model_name, num_labels=num_labels, automodel_args={'ignore_mismatched_sizes': True})
    # Set up the data loader for training
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=train_batch_size)
    # Train the model
    warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1)  # 10% of train data for warm-up
    model.fit(train_dataloader=train_dataloader,
              epochs=num_epochs,
              evaluation_steps=10_000,
              warmup_steps=warmup_steps,
              output_path=model_save_path,
              show_progress_bar=True)

    # Log metrics for the training set
    evaluator_train = CEBinaryClassificationEvaluator(train_examples, name='training')
    accuracy_train = evaluator_train(model)
    mlflow.log_metric("final_accuracy_train", accuracy_train)

    # Validation set
    validation_examples = [InputExample(texts=[entry['tgt'], entry['hyp']], label=int(1 - entry['p(Hallucination)'])) for entry in df_val.to_dict('records')]

    # Evaluate the model on the validation set
    evaluator_val = CEBinaryClassificationEvaluator(validation_examples, name='validation')
    accuracy_val = evaluator_val(model)
    mlflow.log_metric("final_accuracy_val", accuracy_val)

    # Save the trained model explicitly
    model.save(model_save_path)

    # Log the model in MLflow
    mlflow.pytorch.log_model(model, "model")
