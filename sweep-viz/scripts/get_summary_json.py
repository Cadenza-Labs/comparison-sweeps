import os
import pandas as pd
import yaml
import json
import sys
import os
from pathlib import Path

def down_two(sweep_path):
    current_path = sweep_path
    # Loop through the levels, each with only one choice
    for _ in range(2):  # Change this to the number of levels you want to go down
        # Get all directories at the current level
        subdirectories = [entry for entry in current_path.iterdir() if entry.is_dir()]
        
        # Check if there's exactly one directory at this level
        if len(subdirectories) == 1:
            # If there's only one directory at this level, navigate to it
            current_path = subdirectories[0]
        else:
            # Handle the case where there's not exactly one directory at this level
            print(f"Expected one directory at {current_path}, found {len(subdirectories)}.")

    return current_path



def extract_values(eval_filepath, layer_ensembling_filepath):
    eval_df = pd.read_csv(eval_filepath)
    last_layer = eval_df['layer'].max()
    three_quarters_layer = round(last_layer * 0.75)
    eval_conditions = (
        eval_df['layer'].isin([last_layer, three_quarters_layer]) & 
        eval_df['prompt_ensembling'].isin(['full', 'partial', 'none'])
    )
    eval_filtered_df = eval_df[eval_conditions]
    layer_ensembling_df = pd.read_csv(layer_ensembling_filepath)
    layer_ensembling_conditions = layer_ensembling_df['prompt_ensembling'].isin(['full', 'partial', 'none'])
    layer_ensembling_filtered_df = layer_ensembling_df[layer_ensembling_conditions]
    results = {}
    for _, row in eval_filtered_df.iterrows():
        prefix = "last_layer" if row['layer'] == last_layer else "75_layer"
        key = f"{prefix}-{row['prompt_ensembling']}"
        results[key] = row['auroc_estimate']
    for _, row in layer_ensembling_filtered_df.iterrows():
        key = f"layer_ensemble-{row['prompt_ensembling']}"
        results[key] = row['auroc_estimate']
    return results

# Class and function to parse YAML content
class Variant:
    def __init__(self, key, yaml_key, possible_values):
        self.key = key
        self.yaml_key = yaml_key
        self.possible_values = possible_values

    def extract_value(self, yaml_data):
        if not self.yaml_key:
            return None


        nested_keys = self.yaml_key.split('.')
        current_data = yaml_data
        for k in nested_keys:
            if k.startswith("--"):
                k = k[2:]
            if k in current_data:
                current_data = current_data[k]
            else:
                return ""
        if current_data == "elk.training.ccs_reporter.CcsConfig":
            return "ccs"
        if current_data == "elk.training.eigen_reporter.EigenFitterConfig":
            return "eigen"
        if self.key == "norm" and current_data is None:
            return ""
        
        if self.key == "neg_cov_weight":
            # string to 1 dp
            return str(round(current_data, 1))
        
        if isinstance(current_data, list):
            current_data = current_data[0]
        
        return current_data


to_expected = {
    "elk.training.ccs_reporter.CcsConfig": "ccs",
    "elk.training.eigen_reporter.EigenConfig": "eigen",
}

variants = [
    Variant("net", "net._type_", ["ccs", "eigen"]),
    Variant("norm", "net.norm", ["burns", None]),
    Variant("per probe prompt", "probe_per_prompt", ["True", "False"]),
    Variant("neg_cov_weight", "net.neg_cov_weight", [None, 0, 0.5, 1]),
    Variant("loss", "net.loss", ["ccs_prompt_var", None]),
    Variant("erase_prompt", "net.erase_prompts", [False, True]),
]

def parse_yaml_content(yaml_content):
    yaml_data = yaml.safe_load(yaml_content)
    result = {}
    for variant in variants:
        result[variant.key] = variant.extract_value(yaml_data)
    return result

# Main extraction function
def extract_sweep_data_corrected(sweep_path):
    results = {
        "path": os.path.basename(sweep_path),
        "summary": {},
        "parsed_config": {}
    }
    subdir_path = down_two(Path(sweep_path))
    datasets = ['ag_news', 'amazon_polarity', 'dbpedia_14', 'glue:qnli', 'imdb', 'piqa', 'super_glue:boolq', 'super_glue:copa', 'super_glue:rte']
    for dataset in datasets:
        dataset_path = os.path.join(subdir_path, dataset)
        eval_filepath = os.path.join(dataset_path, 'eval.csv')
        layer_ensembling_filepath = os.path.join(dataset_path, 'layer_ensembling.csv')
        yaml_filepath = os.path.join(dataset_path, 'cfg.yaml')
        if not os.path.exists(eval_filepath) or not os.path.exists(layer_ensembling_filepath):
            continue
        results["summary"][dataset] = extract_values(eval_filepath, layer_ensembling_filepath)
        if not results["parsed_config"] and os.path.exists(yaml_filepath):
            with open(yaml_filepath, 'r') as f:
                yaml_content = f.read()
                results["parsed_config"] = parse_yaml_content(yaml_content)
    
    return results

def filter_unique_configs(all_results):
    seen_configs = set()
    unique_results = []

    for result in all_results:
        config = result["parsed_config"]
        config_frozenset = frozenset(config.items())

        if config_frozenset not in seen_configs:
            unique_results.append(result)
            seen_configs.add(config_frozenset)
        else:
            print(f"Duplicate config: {config_frozenset}")
    print(f"removed {len(all_results) - len(unique_results)} duplicates, {len(unique_results)} unique configs")

    return unique_results

def get_summary(sweeps_path):
    all_sweeps = os.listdir(sweeps_path)
    all_results_corrected = [extract_sweep_data_corrected(os.path.join(sweeps_path, sweep)) for sweep in all_sweeps]
    # remove dupicates by config
    filtered = filter_unique_configs(all_results_corrected)
    
    return filtered
