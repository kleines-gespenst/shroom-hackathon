from sentence_transformers import CrossEncoder
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
import pandas as pd

json_file_path = r'D:\Daneshga\TU Wien\Shroom Hackathon\Dataset\SHROOM_validation\val.model-aware.v2.json'

with open(json_file_path, 'r') as file:
    dataset = json.load(file)

# Convert the list of dictionaries to a DataFrame
df = pd.DataFrame(dataset)
df_filtered = df[df['p(Hallucination)'].isin([0, 1])]

# Initialize the CrossEncoder model outside the loop
model = CrossEncoder('vectara/hallucination_evaluation_model')

# Dictionary to store hallucination scores
p_hallucination = {}

# Calculate hallucination scores and store in the dictionary
for index, row in df_filtered.iterrows():
    tgt = row.get("tgt", "Key 'tgt' not found in this entry")
    hyp = row.get("hyp", "Key 'hyp' not found in this entry")

    # Predict hallucination score
    score = model.predict([tgt, hyp])

    # Use the index as the key
    p_hallucination[index] = 1 - score

# Calculate the expression for each row
p_hallucination_values = np.array(list(p_hallucination.values()))
dataset_hallucination_values = np.array(df_filtered["p(Hallucination)"])

# Create a scatter plot
plt.scatter(dataset_hallucination_values, p_hallucination_values, color='blue', alpha=0.5)

# Plot a diagonal line (y = x) for reference
plt.plot([0, 1], [0, 1], color='red', linestyle='--')

# Set axis labels
plt.xlabel('Ground Truth (p(Hallucination) from dataset)')
plt.ylabel('Predicted (p_hallucination)')

# Set plot title
plt.title('Accuracy Illustration')

# Calculate Mean Absolute Error (MAE)
mae = mean_absolute_error(dataset_hallucination_values, p_hallucination_values)

# Print the MAE
print(f'Mean Absolute Error: {mae}')

# Show the scatter plot
plt.show()
