import os
import yaml
from rich import print
import sys
from pathlib import Path

path = Path(sys.argv[1])

# Modify the Variant class and functions to correctly extract neg_cov_weight
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
                return None
        return current_data

# Update the path for neg_cov_weight in the variants list
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


def parse_yaml_content(yaml_content):
    yaml_data = yaml.safe_load(yaml_content)
    result = {}
    for variant in variants:
        result[variant.key] = variant.extract_value(yaml_data)
    return result

def parse_yaml_content_modified(yaml_content):
    """
    Modified parsing function that ensures all list values are converted to tuples for hashability.
    """
    yaml_data = yaml.safe_load(yaml_content)
    result = {}
    for variant in variants:
        value = variant.extract_value(yaml_data)
        # Convert list values to tuples
        if isinstance(value, list):
            value = tuple(value)
        result[variant.key] = value
    return result

def extract_config_from_sweep(sweep_folder):
    """
    Extract the configuration from one of the cfg.yaml files within a consistent sweep.
    """
    for dirpath, _, filenames in os.walk(sweep_folder):
        for filename in filenames:
            if filename == 'cfg.yaml' and "transfer" not in dirpath:
                cfg_path = os.path.join(dirpath, filename)
                return frozenset(parse_yaml_content_modified(open(cfg_path, 'r').read()).items())
            

subfolder_path = os.path.join(path)
subfolder_contents = os.listdir(subfolder_path)

print(subfolder_contents)

def check_configs_within_sweep_exclude_transfer(sweep_folder):
    """
    Check if all non-transfer cfg.yaml files within a sweep have the same configuration.
    """
    # Collect all parsed configurations within the sweep, excluding transfer evaluations
    configurations = [
        frozenset(parse_yaml_content_modified(open(cfg_path, 'r').read()).items()) 
        for cfg_path in [
            os.path.join(dirpath, filename) 
            for dirpath, _, filenames in os.walk(sweep_folder) 
            for filename in filenames if filename == 'cfg.yaml' and "transfer" not in dirpath
        ]
    ]
    
    # If all configurations are the same, their set length should be 1
    return len(set(configurations)) == 1

# Check consistency of configurations within each sweep, excluding transfer evaluations
sweep_consistency_no_transfer = {}
for sweep in subfolder_contents:
    sweep_path = os.path.join(subfolder_path, sweep)
    sweep_consistency_no_transfer[sweep] = check_configs_within_sweep_exclude_transfer(sweep_path)


# Extract parsed configurations for each consistent sweep
sweep_configs = {}
for sweep, is_consistent in sweep_consistency_no_transfer.items():
    if is_consistent:
        sweep_path = os.path.join(subfolder_path, sweep)
        sweep_configs[sweep] = extract_config_from_sweep(sweep_path)

# Check for duplicate configurations between sweeps
duplicates = {}
seen_configs = set()
for sweep, config in sweep_configs.items():
    if config in seen_configs:
        duplicates[sweep] = config
    else:
        seen_configs.add(config)

import collections
print(len(seen_configs))
print(list(collections.Counter(seen_configs).values()))