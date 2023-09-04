import json
import itertools
import os
import sys

# Lists from your previous script
models_list = [
    "meta-llama/Llama-2-7b-hf",
    "meta-llama/Llama-2-13b-hf",
    "EleutherAI/pythia-12b",
    "bigscience/bloom-7b1",
    "EleutherAI/pythia-6.9b",
]

BURNS_DATASETS = [
    "ag_news",
    "amazon_polarity",
    "dbpedia_14",
    "glue:qnli",
    "imdb",
    "piqa",
    "super_glue:boolq",
    "super_glue:copa",
    "super_glue:rte",
]


def generate_config_file_for_combination(model_idx, dataset_idx, GPUS):
    model = models_list[model_idx]
    dataset = BURNS_DATASETS[dataset_idx]
    model_name = model.split("/")[-1]  # Extracting the model name from the path
    filename = f"configs/{dataset}_{model_name}.json"

    # make configs
    os.makedirs("configs", exist_ok=True)

    # Configuration structure
    config = {
        "name": f"{dataset}_{model_name}",
        "GPUS": GPUS,
        "model_indexes": [model_idx],
        "dataset_indexes": [dataset_idx],
    }

    with open(filename, "w") as f:
        json.dump(config, f, indent=4)


if __name__ == "__main__":
    GPUS = 1  # Set to your desired GPU count

    # Generate configurations for every combination of model and dataset
    for model_idx, dataset_idx in itertools.product(
        range(len(models_list)), range(len(BURNS_DATASETS))
    ):
        generate_config_file_for_combination(model_idx, dataset_idx, GPUS)
    print("Configs generated at ./configs")
