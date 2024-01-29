from sentence_transformers import CrossEncoder
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, accuracy_score, f1_score
from scipy.stats import spearmanr
import pandas as pd
import torch

#select GPU
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

print(f"Using GPU is CUDA:{os.environ['CUDA_VISIBLE_DEVICES']}")

for i in range(torch.cuda.device_count()):
    info = torch.cuda.get_device_properties(i)
    print(f"CUDA:{i} {info.name}, {info.total_memory / 1024 ** 2}MB")

# Define the file paths
#json_aware_file_path = 'data_gpt_4_labeled/labeled-test.model-aware.json'
json_agnostic_file_path = 'data_gpt_4_labeled/labeled-test.model-agnostic.json'

# Load and merge the datasets
#with open(json_aware_file_path, 'r') as aware_file:
#    aware_dataset = json.load(aware_file)

with open(json_agnostic_file_path, 'r') as agnostic_file:
   agnostic_dataset = json.load(agnostic_file)

# Merge the datasets
#merged_dataset = aware_dataset + agnostic_dataset

# Convert the list of dictionaries to a DataFrame
#df = pd.DataFrame(merged_dataset)
    
df = pd.DataFrame(agnostic_dataset)

# Initialize the CrossEncoder model outside the loop
model_path = 'models/vectara_finetuned_gpt4labels_mlflow'
model = CrossEncoder(model_path)

# Dictionary to store hallucination scores
p_hallucination = {}

# List to store entries in the desired format
result_entries = []
# Calculate hallucination scores and store in the list
for index, row in df.iterrows():
    task = row.get("task", "Key 'task' not found in this entry")
    tgt = row.get("tgt", "Key 'tgt' not found in this entry")
    hyp = row.get("hyp", "Key 'hyp' not found in this entry")
    
    if task == "PG":
        src = row.get("src", "Key 'src' not found in this entry")
        tgt = src  # Use 'src' instead of 'tgt'

    # Predict hallucination score
    score = model.predict([tgt, hyp])
    score = 1 - score

    # Use the index as the key
    p_hallucination[index] = score

    # Determine label based on threshold
    if score > 0.5:
        label = "Hallucination"
    else:
        label = "Not Hallucination"

    # Create an entry in the desired format
    entry = {
        'id': row['id'],
        'p(Hallucination)': score,
        'label': label
    }
    
    result_entries.append(entry)

# Calculate the expression for each row
p_hallucination_values = np.array(list(p_hallucination.values()))
dataset_hallucination_values = np.array(df["p(Hallucination)"])

labels_predicted = []
for el in p_hallucination_values:
    labels_predicted.append(int(el>0.5))

labels_gpt4 = []
for el in dataset_hallucination_values:
    labels_gpt4.append(int(el>0.5))



# Calculate Spearman's Rank Correlation Coefficient
rho, _ = spearmanr(dataset_hallucination_values, p_hallucination_values)

# Create a scatter plot
plt.scatter(dataset_hallucination_values, p_hallucination_values, color='blue', alpha=0.5)

# Plot a diagonal line (y = x) for reference
plt.plot([0, 1], [0, 1], color='red', linestyle='--')

# Set axis labels
plt.xlabel('Ground Truth (p(Hallucination) from dataset)')
plt.ylabel('Predicted (p_hallucination)')

# Set plot title
plt.title('Accuracy Illustration')

# Print the Spearman's Rank Correlation Coefficient
print(f'Spearman\'s Rank Correlation Coefficient (rho): {rho}')

# Calculate Mean Absolute Error (MAE)
mae = mean_absolute_error(dataset_hallucination_values, p_hallucination_values)

# Print the MAE
print(f'Mean Absolute Error: {mae}')

# Calculate Mean Absolute Error (MAE)
acc = accuracy_score(labels_gpt4, labels_predicted)

# Print the MAE
print(f'Accuracy Score: {acc}')

# Calculate Mean Absolute Error (MAE)
f_score = f1_score(labels_gpt4, labels_predicted)

# Print the MAE
print(f'f1-score: {f_score}')

# Show the scatter plot
plt.show()

# Save the list of entries to a new JSON file
output_json_path = "results/vectara_gpt4_finetuned_on_training_set/test.model-agnostic.json"
with open(output_json_path, 'w') as output_file:
    json.dump(result_entries, output_file, indent=2)