import os
import pandas as pd
import yaml
import os
from pathlib import Path
from rich import print

from dataclasses import dataclass

@dataclass
class Sweep:
    name: str | None
    models: list
    datasets: list
    args: dict

    def from_df_row(row: pd.Series):
        model = row['model']
        dataset = row['dataset']
        columns_to_drop =  ['sweep_name', 'model', 'dataset']
        columns_to_drop_existing = [col for col in columns_to_drop if col in row]
        row = row.drop(columns_to_drop_existing)
        # if value is NAN OR empty then drop the column
        # row = row.dropna()
        row = row[row != ""]
        return Sweep(
            name=row.get('sweep_name', None),
            models=[model],
            datasets=[dataset],
            args=row.to_dict()
        )
    
    def to_df_row(self):
        return {
            'sweep_name': self.name,
            'models': ' '.join(self.models),
            'datasets': ' '.join(self.datasets),
            'args': self.args
        }

    def to_command(self, extra_args={}):
        argstring = " ".join([f"--{k} {v}" for k, v in (*self.args.items(), *extra_args.items())])
        return f"elk sweep --models {' '.join(self.models)} --datasets {' '.join(self.datasets)} {argstring}"

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
            if current_path.name == "gpt2":
                return current_path
        else:
            # Handle the case where there's not exactly one directory at this level
            print(f"Expected one directory at {current_path}, found {len(subdirectories)}.")

    return current_path



def extract_values(eval_filepath, layer_ensembling_filepath):
    # print(f"extracting values from {eval_filepath} and {layer_ensembling_filepath}")
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

df = pd.DataFrame()

# Main extraction function
def extract_sweep_data_corrected(sweep_path):
    local_df = pd.DataFrame()
    results = {
        "sweep_name": os.path.basename(sweep_path),
        "summary": {},
        "parsed_config": {}
    }
    sweep_path = Path(sweep_path)

    # walk through sweep_path. if it has eval.csv and layer_ensembling.csv, then it is a dataset_path. collect all of them.
    dataset_paths = []
    for root, dirs, files in os.walk(sweep_path):
        if 'eval.csv' in files and 'layer_ensembling.csv' in files and 'transfer' not in root:
            dataset_paths.append(Path(root))


    for dataset_path in dataset_paths:  # for each dataset path, we want to get the eval.csv and layer_ensembling.csv and dump into a df and append
        dataset = dataset_path.name
        eval_filepath = os.path.join(dataset_path, 'eval.csv')
        layer_ensembling_filepath = os.path.join(dataset_path, 'layer_ensembling.csv')
        yaml_filepath = os.path.join(dataset_path, 'cfg.yaml')
        if not os.path.exists(eval_filepath) or not os.path.exists(layer_ensembling_filepath):
            continue
        results["summary"] = extract_values(eval_filepath, layer_ensembling_filepath)
        if not results["parsed_config"] and os.path.exists(yaml_filepath):
            with open(yaml_filepath, 'r') as f:
                yaml_content = f.read()
                results["parsed_config"] = parse_yaml_content(yaml_content)
        # also get: model and ds, should be easy by just printing from the path
        # i guess we also want the transfer configs too
        # and robust error handling, list of sweeps that don't have all the right stuff
        # read config yaml
        config = yaml.safe_load(open(yaml_filepath, 'r'))
        model = config['data']['model']
        
        results["model"] = model
        results["dataset"] = dataset

        flattened_res = {
            **results['parsed_config'], 
            **{'model': model, 'dataset': dataset, 'sweep_name': results['sweep_name']}, 
            **results['summary']
        }
        local_df = pd.concat([local_df, pd.DataFrame.from_dict([flattened_res])], ignore_index=True)
    return local_df

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
    all_results_corrected = pd.concat([extract_sweep_data_corrected(os.path.join(sweeps_path, sweep)) for sweep in all_sweeps])
    # remove dupicates by config
    # filtered = filter_unique_configs(all_results_corrected)
    
    return all_results_corrected

expected_combos = {'eigen--True-0.5--True', 'eigen--False-1.0--False', 'eigen--True-0.0--False', 'ccs-burns-False--1.0*ccs-True', 'eigen--False-0.0--False', 'eigen--False-0.0--True', 'eigen--False-0.5--False',
'ccs-leace-False--1.0*ccs_prompt_var-True', 'ccs-leace-True--1.0*ccs-True', 'eigen--True-0.5--False', 'ccs-burns-False--1.0*ccs_prompt_var-True', 'eigen--False-0.5--True', 'ccs-burns-True--1.0*ccs-True', 'ccs-burns-True--1.0*ccs_prompt_var-True',
'ccs-leace-False--1.0*ccs-True', 'eigen--False-1.0--True', 'ccs-leace-True--1.0*ccs_prompt_var-True', 'eigen--True-1.0--False', 'eigen--True-1.0--True', 'eigen--True-0.0--True'}

# get combos from specifically llama 13b
def filter_combos(df):
    # filter only expected combos but drop them before returning
    print(f'dropping {len(df[~df["combo"].isin(expected_combos)])} rows')
    return df[df['combo'].isin(expected_combos)]

def filter_models(df):
    expected_models = [
        # 'allenai/unifiedqa-t5-11b',
        'EleutherAI/gpt-j-6b',
        # 'bigscience/T0pp',
        'bigscience/bloom-7b1',
        # 'huggyllama/llama-7b',
        'EleutherAI/pythia-6.9b',
        # 'huggyllama/llama-30b',
        'meta-llama/Llama-2-13b-hf',
        # 'EleutherAI/pythia-12b',
        # 'sshleifer/tiny-gpt2',
        'meta-llama/Llama-2-7b-hf',
        # 'microsoft/deberta-v2-xxlarge-mnli'
    ]
    print(f'dropping {len(df[~df["model"].isin(expected_models)])} rows')
    return df[df['model'].isin(expected_models)]

def find_missing_combinations(df):
    # Create a column with the combination of the six config values as a string
    print(df)
    df['combo'] = df[['net', 'norm', 'per probe prompt', 'neg_cov_weight', 'loss', 'erase_prompt']].astype(str).apply(lambda x: '-'.join(x), axis=1)

    # filter only expected combos
    df = filter_combos(df)

    # Create a DataFrame of all possible combinations
    all_possible_combinations = pd.MultiIndex.from_product([
        df['model'].unique(),  # models
        df['dataset'].unique(),  # 9
        pd.Series(list(expected_combos))  # 20
    ], names=['model', 'dataset', 'combo']).to_frame(index=False)

    # Merge with unique_entries to find missing combinations
    unique_entries = df[['model', 'dataset', 'combo']].drop_duplicates()
    missing_combinations = all_possible_combinations.merge(unique_entries, on=['model', 'dataset', 'combo'], how='left', indicator=True)
    missing_combinations = missing_combinations[missing_combinations['_merge'] == 'left_only'][['model', 'dataset', 'combo']]
    
    # Split the 'combo' column back to original columns
    config_cols = ['net', 'norm', 'per probe prompt', 'neg_cov_weight', 'loss', 'erase_prompt']
    if len(missing_combinations) == 0:
        print("No missing combinations found")
        return []
    else:
        missing_combinations[config_cols] = missing_combinations['combo'].str.split('-', expand=True)
        
        # Convert "nan" back to empty strings for certain columns
        for col in config_cols:
            missing_combinations[col] = missing_combinations[col].replace('nan', '')
        
        # Drop the 'combo' column and return the missing rows

        print(f"{len(missing_combinations)} missing combinations found")
        # print breakdown count by model
        print(missing_combinations['model'].value_counts())
        # print breakdown count by dataset
        print(missing_combinations['dataset'].value_counts())
        # print breakdown count by combo
        print(missing_combinations['combo'].value_counts())

        return [Sweep.from_df_row(row) for i, row in missing_combinations.iterrows()]


def add_combo(df):
    df['combo'] = df[['net', 'norm', 'per probe prompt', 'neg_cov_weight', 'loss', 'erase_prompt']].astype(str).apply(lambda x: '-'.join(x), axis=1)
    return df

if __name__ == '__main__':
    import os
    from pathlib import Path

    filename = 'all_results.csv'

    # load if exists
    if False:
        df = pd.read_csv(filename)
    else:
        data_directory = Path('./data').resolve()  # Replace with the correct path to your data directory
        all_sweep_dirs = [
            (data_directory / sweeps).resolve() 
            for sweeps in os.listdir(data_directory) 
            if (path := Path(os.path.join(data_directory, sweeps))).is_dir() and 
            path.name.endswith('_no_reporters')
        ]
        for sweeps in all_sweep_dirs:
            summary_df = get_summary(sweeps)
            add_combo(summary_df)
            print(f"{sweeps.name} has {len(summary_df)} rows")
            df = pd.concat([df, summary_df], ignore_index=True)
            # if '7b' in sweeps.name:
            #     breakpoint()
            # how many duplicates in summary_df
            # print(f"Duplicate rows: {len(summary_df[summary_df.duplicated()])}")

    df['combo'] = df[['net', 'norm', 'per probe prompt', 'neg_cov_weight', 'loss', 'erase_prompt']].astype(str).apply(lambda x: '-'.join(x), axis=1)
    df = filter_combos(df)
    df = filter_models(df)
    df = df.drop_duplicates(subset=['model', 'dataset', 'combo'])

    df.to_csv(filename, index=False)
    missing_sweeps = find_missing_combinations(df)
    print(f"Unique models: {df['model'].unique()}, {len(df['model'].unique())}")
    # make into a python list df['model'].unique()

    print(f"Unique datasets: {df['dataset'].unique()}, {len(df['dataset'].unique())}")
    print(f"Unique combos: {df['combo'].unique()}, {len(df['combo'].unique())}")

    # check for duplicate rows
    # df_sorted = df.sort_values(by=['model', 'dataset', 'combo'])
    # duplicate_rows = df[df[['model', 'dataset', 'combo']].duplicated(keep=False)]
    # duplicate_rows.to_csv('duplicate_rows.csv', index=False)
    # print(duplicate_rows)

    # drop duplicate rows by model, dataset, combo
    

    extra_args = {
        'num_gpus': 4,
    }
    for sweep in missing_sweeps[:10]:
        print(f'"{sweep.to_command(extra_args)}" \\')
        

    