from pathlib import Path
from typing import Dict, List
import pandas as pd
import re
from rich import print
from collections import defaultdict
import json
import numpy as np
import seaborn as sns
from bs4 import BeautifulSoup
import os

LOG_FILES = "/Users/jon/llama7b_logs.txt"
old_root_dir = Path("/data/jonathan_ng/elk-reporters/sweeps")
new_root_dir = Path("/Users/jon/sweeps-eval-only")
score_type = 'auroc_estimate'


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
        results[key] = row[score_type]
        
    for _, row in layer_ensembling_filtered_df.iterrows():
        key = f"layer_ensemble-{row['prompt_ensembling']}"
        results[key] = row[score_type]
        
    return results

def get_all_subfolders(path):
    subfolders = []
    for dirpath, dirnames, filenames in os.walk(path):
        for dirname in dirnames:
            subfolders.append(Path(os.path.join(dirpath, dirname)))
    return subfolders

def sweep_path_to_summary(old_root: Path, new_root: Path, unique_triples_sorted): # -> Dict[Path, List[Summary]]:
    """
    Replace old sweep paths with new sweep paths and extract summaries from sweep folders.

    Args:
    - old_root (Path): Path to the old root directory.
    - new_root (Path): Path to the new root directory.

    Returns:
    - Dict[Path, List[Summary]]: Dictionary mapping sweep paths to lists of Summary objects.
    """
    # Replace old sweep paths with new sweep paths
    old_root_str = str(old_root)
    sweep_paths = [path.replace(old_root_str, str(new_root)) for _, _, path in unique_triples_sorted]

    summaries_by_sweep = {}

    # Walk through each sweep folder
    for sweep_path in sweep_paths:
        summaries = defaultdict(list)
        for subfolder in get_all_subfolders(sweep_path):
            eval_file = subfolder / "eval.csv"
            layer_ensembling_file = subfolder / "layer_ensembling.csv"

            # Check if both files exist in the sweep folder
            if eval_file.exists() and layer_ensembling_file.exists():

                # Extract summaries using the provided function
                summary = extract_values(eval_file, layer_ensembling_file)
                summaries[subfolder.name].append(summary)
        summaries_by_sweep[sweep_path] = summaries

    return summaries_by_sweep

def extract_unique_commands_and_sweeps(log_file_path: str): # -> List[Tuple[str, str, str]]:
    """
    Extracts unique commands, sweep names, and sweep paths from the provided log file.
    
    Args:
    - log_file_path (str): Path to the log file.
    
    Returns:
    - List[Tuple[str, str, str]]: List of unique tuples where each tuple contains a command, its corresponding sweep name, 
      and the sweep path.
    """
    with open(log_file_path, "r") as file:
        log_contents = file.readlines()

    # Regular expressions to capture the desired command and sweep name structures
    command_patterns = [
        r"Running command: (elk sweep .+)",
        r"Sweep \[\d+\]: (elk sweep .+)"
    ]
    
    # Modified regex to capture the full path after the escape sequence
    sweep_path_regex = r"Saving sweep results to \x1b\[1m(/data/jonathan_ng/elk-reporters/sweeps/([\w\-]+))\x1b\[0m"

    triples = []

    # Iterate over the log_contents to pair commands with their respective paths and sweep names
    for idx, line in enumerate(log_contents):
        for pattern in command_patterns:
            command_match = re.search(pattern, line)
            if command_match:
                command = command_match.group(1)
                
                # Look ahead in the next lines to find the associated sweep path for the current command
                for lookahead_line in log_contents[idx+1:idx+10]:  # Assuming path is within next 10 lines for safety
                    path_match = re.search(sweep_path_regex, lookahead_line)
                    if path_match:
                        path = path_match.group(1)
                        name = path_match.group(2)
                        triples.append((command, name, path))
                        break  # Break once we find the associated path

    # Filter for unique triplets
    unique_triples = list(set(triples))
    
    return unique_triples

# Extracting unique triplets from the log file
unique_triples_sorted = sorted(extract_unique_commands_and_sweeps(LOG_FILES), key=lambda x: (x[1], x[0]))
# Counting the number of unique triplets
num_unique_triples = len(unique_triples_sorted)

sweep_to_summaries = sweep_path_to_summary(old_root_dir, new_root_dir, unique_triples_sorted)

import pandas as pd

def aggregate_summary(summaries):
    """
    Aggregates the list of summaries by calculating the mean of all respective values using pandas.
    
    Args:
    - summaries (list): List of summaries where each summary is a dictionary of key-values.
    
    Returns:
    - dict: Aggregated summary.
    """
    df = pd.DataFrame(summaries)
    return df.mean().to_dict()

def create_corrected_summary_dict(triples, summaries_data, old_root, new_root):
    """
    Create a summary dictionary with corrected paths.
    """
    summary_dict = {}
    
    for command, sweep_name, sweep_path in triples:
        # Convert to new path format
        new_path = sweep_path.replace(str(old_root), str(new_root))
        
        if new_path in summaries_data:
            summaries = []
            for dataset_summaries in summaries_data[new_path].values():
                summaries.extend(dataset_summaries)

            agg_summary = aggregate_summary(summaries)
            #(command, new_path, agg_summary)
            summary_dict[sweep_name] = {
                "command": command,
                "path": new_path,
                "summary": agg_summary
            }

    return summary_dict

# Define the old and new root directories
old_root_dir_str = "/data/jonathan_ng/elk-reporters/sweeps"
new_root_dir_str = "/Users/jon/sweeps-eval-only"

# Generate the summary dictionary using the corrected function
corrected_summary_dict = create_corrected_summary_dict(unique_triples_sorted, sweep_to_summaries, old_root_dir_str, new_root_dir_str)



from typing import List, Union, NamedTuple

class Variant(NamedTuple):
    name: str
    arg: str
    values: List[Union[str, None, int, float]]



def improved_parse_command(command: str) -> Dict[str, Union[str, None, int, float]]:
    """
    Improved parse function that prioritizes exact matches over partial matches.
    
    Args:
    - command (str): Command string to parse.
    
    Returns:
    - dict: Dictionary containing the values of the relevant arguments.
    """
    variants = [
        Variant("net", "--net", ["ccs", "eigen"]),
        Variant("norm", "--norm", ["burns", None]),
        Variant("per probe prompt", "--probe_per_prompt", ["True", "False"]),
        Variant("prompt indices", "--prompt_indices", ["1", None]),
        Variant("neg_cov_weight", "--neg_cov_weight", [0, 0.5, 1, None]),
        Variant("loss", "--loss", ["ccs_prompt_var", None])
    ]

    parsed_values = {}

    for variant in variants:
        value = None
        # Sorting values to prioritize exact matches (longer strings) first
        sorted_values = sorted(variant.values, key=lambda x: 0 if x is None else len(str(x)), reverse=True)
        for potential_value in sorted_values:
            if potential_value is None:
                if variant.arg not in command:
                    value = None
                    break
            elif f"{variant.arg}={potential_value}" in command or f"{variant.arg} {potential_value}" in command:
                value = potential_value
                break
        parsed_values[variant.name] = value

    return parsed_values

commands = [triplet[0] for triplet in unique_triples_sorted]

# #  add parsed commands to the summary dictionary
for d in corrected_summary_dict.values():
    d['parsed_command'] = improved_parse_command(d['command'])


# with open("unique_triples_sorted.json", "w") as file:
#     json.dump(unique_triples_sorted, file)

# with open("sweep_to_summaries.json", "w") as file:
#     json.dump(sweep_to_summaries, file)

with open("corrected_summary_dict.json", "w") as file:
    json.dump(corrected_summary_dict, file, indent=4)

with open("corrected_summary_dict.json", 'r') as file:
    data = json.load(file)

# Convert the data into a pandas DataFrame
df = pd.DataFrame(data)

# Extract the 'summary' and 'parsed_command' rows for easier processing
summary_data = df.loc['summary'].to_dict()
parsed_command_data = df.loc['parsed_command'].to_dict()

# Create a new DataFrame to hold the reshaped data
reshaped_data = []

for key, value in summary_data.items():
    temp_data = parsed_command_data[key]
    temp_data.update(value)
    reshaped_data.append(temp_data)

# Convert the reshaped data into a DataFrame
extended_df = pd.DataFrame(reshaped_data)

# Define non-value columns and value columns
non_value_columns = ['net', 'norm', 'prompt indices', 'neg_cov_weight', 'loss', 'per probe prompt']
value_columns = [col for col in extended_df.columns if '75_layer' in col or 'last_layer' in col or 'layer_ensemble' in col]

# Create the combined DataFrame with value columns and label descriptions
final_combined_df = pd.concat([extended_df[non_value_columns], extended_df[value_columns]], axis=1)

# for the last nine columns in the DataFrame, replace the float with max(x, 1 - x)
corrected_df = final_combined_df.copy()
for col in value_columns:
    corrected_df[col] = corrected_df[col].apply(lambda x: max(x, 1 - x))

# Convert the DataFrame into an HTML table
html_table = corrected_df.to_html(classes='heatmap', escape=False, table_id="heatmapTable")

# Define a basic CSS for styling the heatmap
heatmap_css = """
<style>
    .heatmap {
        border-collapse: collapse;
        width: 100%;
        font-family: Arial, sans-serif;
    }
    .heatmap th, .heatmap td {
        border: 1px solid #d4d4d4;
        padding: 8px 12px;
    }
    .heatmap th {
        background-color: #f2f2f2;
    }
    .heatmap td {
        text-align: center;
    }
    /* You can add more styles or JavaScript for interactivity if needed */
</style>
"""

# Combine the CSS and the HTML table
html_content = heatmap_css + html_table

# Save the HTML content to a file
html_filename = "heatmap_table.html"
with open(html_filename, 'w') as file:
    file.write(html_content)

# Function to convert RGB values in range [0, 1] to hexadecimal
def rgb_to_hex(rgb):
    return '#{:02x}{:02x}{:02x}'.format(int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))


def apply_heatmap_to_html(input_path, output_path):
    """
    Applies the coolwarm color scheme to the last 9 columns of the table in the given HTML file.
    
    Parameters:
    - input_path (str): Path to the input HTML file.
    - output_path (str): Path to save the modified HTML file.
    
    Returns:
    - None
    """
    
    # Load the HTML content
    with open(input_path, "r") as file:
        html_content = file.read()

    # Parse the HTML using BeautifulSoup
    soup = BeautifulSoup(html_content, 'html.parser')

    # Extract the table and the data values from the last 9 columns
    table = soup.find("table")
    rows = table.find_all("tr")[1:]  # Exclude header
    data_values = [[float(cell.text) if cell.text.lower() != 'nan' else np.nan for cell in row.find_all("td")[-9:]] for row in rows]

    # Normalize the data values
    flat_data = np.array(data_values).flatten()
    valid_data = flat_data[~np.isnan(flat_data)]
    data_min = valid_data.min()
    data_max = valid_data.max()
    normalized_values = (np.array(data_values) - data_min) / (data_max - data_min)



    # Map normalized values to the coolwarm color scheme
    colors = sns.color_palette("coolwarm", as_cmap=True)(normalized_values)
    hex_colors = np.array([[rgb_to_hex(cell) for cell in row] for row in colors])

    # Apply the colors to the table cells
    for i, row in enumerate(rows):
        for j, cell in enumerate(row.find_all("td")[-9:]):
            cell["style"] = f"background-color: {hex_colors[i][j]}"

    # Save the modified content
    with open(output_path, "w") as file:
        file.write(str(soup))
    
    print(f"Modified HTML saved to {output_path}")

# Test the function
test_input_path = "heatmap_table.html"
test_output_path = "test_modified_heatmap_table.html"
apply_heatmap_to_html(test_input_path, test_output_path)
