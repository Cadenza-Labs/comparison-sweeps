import argparse
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import rich

# Set up argument parser
parser = argparse.ArgumentParser(description='Process eval.csv files and display a heatmap.')
parser.add_argument('path', type=str, help='Path to the meta-llama directory.')

# Retrieve arguments
args = parser.parse_args()
model_path = args.path

# Function to search for specific files within a directory.
def search_files(directory, file_name):
    print(directory)
    paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file == file_name:
                paths.append(os.path.join(root, file))
    return paths

# Search for eval.csv files
eval_csv_paths = search_files(os.path.join(model_path), "eval.csv")
print(f"Found {len(eval_csv_paths)} eval.csv files.")

# Filter out eval.csv paths that are not inside the transfer folder
main_eval_csv_paths = [path for path in eval_csv_paths if "transfer" not in path]
print(f"Found {len(main_eval_csv_paths)} eval.csv files in the main folder.")

# Extract values function
def extract_values(eval_filepath, layer_ensembling_filepath):
    """
    Extracts the required values from the provided CSV files and returns them as a dictionary.

    Args:
    - eval_filepath (str): Path to the eval CSV file.
    - layer_ensembling_filepath (str): Path to the layer ensembling CSV file.

    Returns:
    - dict: Dictionary containing the extracted values.
    """
    # Read the eval CSV file
    eval_df = pd.read_csv(eval_filepath)
    # Determine the last layer
    last_layer = eval_df['layer'].max()
    # Calculate the layer corresponding to round(last_layer * 0.75)
    three_quarters_layer = round(last_layer * 0.75)

    # Filter rows for the specific layers and prompt_ensembling values
    eval_conditions = (
        eval_df['layer'].isin([last_layer, three_quarters_layer]) & 
        eval_df['prompt_ensembling'].isin(['full', 'partial', 'none'])
    )
    eval_filtered_df = eval_df[eval_conditions]

    # Read the layer ensembling CSV file
    layer_ensembling_df = pd.read_csv(layer_ensembling_filepath)
    # Filter rows for the specified prompt_ensembling values
    layer_ensembling_conditions = layer_ensembling_df['prompt_ensembling'].isin(['full', 'partial', 'none'])
    layer_ensembling_filtered_df = layer_ensembling_df[layer_ensembling_conditions]

    # Combine results
    results = {}

    for _, row in eval_filtered_df.iterrows():
        prefix = "last_layer" if row['layer'] == last_layer else "75_layer"
        key = f"{prefix}-{row['prompt_ensembling']}"
        results[key] = row['auroc_estimate']
        
    for _, row in layer_ensembling_filtered_df.iterrows():
        key = f"layer_ensemble-{row['prompt_ensembling']}"
        results[key] = row['auroc_estimate']
        
    return results

# Extract values for each of the main datasets
datasets_results = {}
for path in main_eval_csv_paths:
    dataset_name = path.split("/")[-2]
    datasets_results[dataset_name] = extract_values(path, path.replace("eval.csv", "layer_ensembling.csv"))

# Convert the results into a DataFrame
results_df = pd.DataFrame(datasets_results).transpose()
rich.print(results_df)

# Compute the mean over datasets and add it to the leftmost column
results_df["mean"] = results_df.mean(axis=1)
results_df = results_df[["mean"] + [col for col in results_df.columns if col != "mean"]]

# Display heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(results_df, annot=True, cmap="YlGnBu", linewidths=.5, fmt=".4f")
plt.title("Extracted Values Heatmap")
plt.show()
