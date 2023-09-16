import os
import pandas as pd
import yaml
import json
import sys
import os
from pathlib import Path
import re
from rich import print
from render_summary_json import render_summary_json

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

def parse_yaml_content(yaml_path):
    with open(yaml_path, 'r') as f:
        yaml_content = f.read()
    yaml_data = yaml.safe_load(yaml_content)
    result = {}
    for variant in variants:
        result[variant.key] = variant.extract_value(yaml_data)
    return result


def extract_multiple_sweep_paths_from_log(file_path, limit=20):
    """
    Extract multiple sweep paths from the log file up to a given limit.
    
    Args:
    - file_path (str): Path to the log file.
    - limit (int): Maximum number of paths to extract.
    
    Returns:
    - list: List of extracted sweep paths.
    """
    paths = []
    with open(file_path, 'r', encoding='ISO-8859-1') as file:
        for line in file.readlines():
            if "Saving sweep results to" in line:
                result_path = re.search(r"Saving sweep results to \x1b\[1m(.*?)\x1b\[0m", line)
                if result_path:
                    paths.append(result_path.group(1))
                if len(paths) == limit:
                    break
    return paths

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


new_root = Path('/Users/jon/projects/notodai/scripts/comparison-sweeps/sweep-viz/data/sweeps-no')
sweep_paths = [new_root / Path(sweep).name for i in range(26) for sweep in extract_multiple_sweep_paths_from_log(f"/Users/jon/Downloads/scripts-230906-beta/not-133-sweep-out-{i}.txt")]
second = [new_root / Path(sweep).name for i in range(9) for sweep in extract_multiple_sweep_paths_from_log(f"/Users/jon/Downloads/scripts-230912/not-133-sweep-out-{i}.txt")]
MISSING = ""

sweep_paths += second + [MISSING]
assert len(sweep_paths) == 520 + 180
assert len(set(sweep_paths)) == 520 + 180

def hashconfig(config):
    return frozenset(config.items())

def get_ds_dir(sweep_path):
    one = sweep_path.iterdir().__next__()
    if one.name == "gpt2":
        ds = one.iterdir().__next__()
        return ds
    else:
        ds = one.iterdir().__next__().iterdir().__next__()
        return ds
    
datasets = {'amazon_polarity', 'imdb', 'ag_news', 'piqa', 'super_glue:copa', 'glue:qnli', 'super_glue:rte', 'dbpedia_14', 'super_glue:boolq'}

assert len(set([get_ds_dir(sweep_path).name for sweep_path in sweep_paths])) == 9

all_parsed = set()
all_models = set()
all_datasets = set()

for sweep_path in sweep_paths:
    
    ds = get_ds_dir(sweep_path)
    all_datasets.add(ds.name)
    if ds.parent.name == "gpt2":
        model = "gpt2"
    else:
        model = f"{ds.parent.parent.name}/{ds.parent.name}"

    all_models.add(model)

    config = ds / "cfg.yaml"
    parsed  = parse_yaml_content(config)
    all_parsed.add(hashconfig(parsed))


assert len(all_parsed) == 20


print(all_models)
print(all_datasets)

import itertools
all_combos = set(itertools.product(all_models, all_datasets, all_parsed))
print(len(all_combos))

def get_parsed_config(sweep_path):
    ds = get_ds_dir(sweep_path)
    if ds.parent.name == "gpt2":
        model = "gpt2"
    else:
        model = f"{ds.parent.parent.name}/{ds.parent.name}"
    config = ds / "cfg.yaml"
    parsed  = parse_yaml_content(config)
    return parsed

def get_model(sweep_path):
    ds = get_ds_dir(sweep_path)
    if ds.parent.name == "gpt2":
        model = "gpt2"
    else:
        model = f"{ds.parent.parent.name}/{ds.parent.name}"
    return model

# for sweep in sweep_paths:
#     parsed = get_parsed_config(sweep)
#     model = get_model(sweep)
#     all_combos.remove((model, ds.name, hashconfig(parsed)))

# print(all_combos)
# print(len(all_combos))


all_sweep_paths = Path("/Users/jon/projects/notodai/scripts/comparison-sweeps/sweep-viz/data/sweeps-no").iterdir()
# for sweep in all_sweep_paths:
#     ds_dir = get_ds_dir(sweep)
#     if ds_dir.parent.name == "gpt2":
#         print(ds_dir)

gpt2_ds_dirs = {get_ds_dir(sweep) for sweep in all_sweep_paths if get_ds_dir(sweep).parent.name == "gpt2" and get_ds_dir(sweep).name == "imdb"}
# print(gpt2_ds_dirs)

needed = all_parsed.copy()

gpt2_sweeps = []
for sweep in gpt2_ds_dirs:
    config = sweep / "cfg.yaml"
    parsed  = parse_yaml_content(config)
    if hashconfig(parsed) in needed:
        gpt2_sweeps.append(sweep.parent.parent)
        needed.remove(hashconfig(parsed))

all_sweeps = gpt2_sweeps + sweep_paths

def extract_sweep_data_corrected(dataset_paths):
    results = {
        "summary": {},
        "parsed_config": {}
    }
    for dataset_path in dataset_paths:
        dataset = dataset_path.name
        eval_filepath = os.path.join(dataset_path, 'eval.csv')
        layer_ensembling_filepath = os.path.join(dataset_path, 'layer_ensembling.csv')
        yaml_filepath = os.path.join(dataset_path, 'cfg.yaml')
        if not os.path.exists(eval_filepath) or not os.path.exists(layer_ensembling_filepath):
            continue
        results["summary"][dataset] = extract_values(eval_filepath, layer_ensembling_filepath)
        if not results["parsed_config"] and os.path.exists(yaml_filepath):
            results["parsed_config"] = parse_yaml_content(yaml_filepath)
    
    return results


def get_summary(config, model):
    sweeps = [sweep for sweep in all_sweeps if hashconfig(get_parsed_config(sweep)) == config]
    # partition by model
    sweeps_by_model = {}
    for sweep in sweeps:
        model = get_model(sweep)
        if model not in sweeps_by_model:
            sweeps_by_model[model] = []
        sweeps_by_model[model].append(sweep)


    # get summary for each model
    summary_by_model = {}
    for model, sweeps in sweeps_by_model.items():
        ds_dirs = [get_ds_dir(sweep) for sweep in sweeps]
        summary_by_model[model] = extract_sweep_data_corrected(ds_dirs)

    summary = summary_by_model[model]

    # with open(f'./data/summary_{model}.json', 'w') as f:
    #     json.dump(summary, f, indent=4)
    return summary

for model in all_models:
    model = model.replace("/", "-")
    summaries = [get_summary(config, model) for config in all_parsed]
    print(summaries)
    with open(f'./data/summary_{model}.json', 'w') as f:
        json.dump(summaries, f, indent=4)
    render_summary_json(summaries, model)

