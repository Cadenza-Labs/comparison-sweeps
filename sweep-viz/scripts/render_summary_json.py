import pandas as pd
import numpy as np
import matplotlib as mpl
from bs4 import BeautifulSoup
import json
import seaborn as sns
from pathlib import Path

def render_summary_json(data, model_name):

    # # Load the JSON data
    # with open("results.json", "r") as file:
    #     json = json.load(file)

    # Preprocess the data into a DataFrame
    def generate_table_for_sweep(sweep):
        summary = sweep["summary"]
        parsed_config = sweep["parsed_config"]
        flattened_summary = {}
        for command in summary["ag_news"]:
            flattened_summary[f"{command}"] = 0
        for _, commands in summary.items():
            for command, value in commands.items():
                flattened_summary[f"{command}"] += value

        flattened_summary = {k: v / len(summary) for k, v in flattened_summary.items()}
        combined_data = {**flattened_summary, **parsed_config}
        return pd.DataFrame([combined_data])

    all_sweep_tables = [generate_table_for_sweep(sweep) for sweep in data]
    combined_table = pd.concat(all_sweep_tables, ignore_index=True)

    # Calculate the mean values across datasets for each configuration
    configurations = [
        '75_layer-full', '75_layer-none', '75_layer-partial',
        'last_layer-full', 'last_layer-none', 'last_layer-partial',
        'layer_ensemble-full', 'layer_ensemble-partial', 'layer_ensemble-none'
    ]
    # Calculate the mean for each configuration across all datasets
    mean_values = {}
    for config in configurations:
        # Get all columns related to the current configuration
        config_cols = [col for col in combined_table.columns if config in col]
        
        # Calculate the mean across the datasets for this configuration
        mean_values[config] = combined_table[config_cols].mean(axis=1)

    # Convert the mean values to a DataFrame
    mean_df_all_configs = pd.DataFrame(mean_values)
    parsed_config_cols = list(data[0]["parsed_config"].keys())
    # Concatenate the mean values with the parsed_config columns
    full_mean_df = pd.concat([mean_df_all_configs, combined_table[parsed_config_cols]], axis=1)

    min_value = mean_df_all_configs[configurations].min().min()
    max_value = mean_df_all_configs[configurations].max().max()

    # Define custom colormap based on the data range
    def custom_color_map_v3(val):
        normalized_val = np.clip((val - min_value) / (max_value - min_value), 0, 1)
        lower_rgb = np.array([62, 127, 147])
        higher_rgb = np.array([195, 85, 58])
        color = np.clip((1 - normalized_val) * lower_rgb + normalized_val * higher_rgb, 0, 255)
        return f'background-color: {mpl.colors.rgb2hex(color/255)};'

    # Apply the custom colormap to the selected columns
    styled_df = full_mean_df.style.map(lambda val: custom_color_map_v3(val) if isinstance(val, (int, float)) else "", subset=configurations)
    html = styled_df.to_html()
    # add model name
    # Create a BeautifulSoup object with the desired HTML structure
    soup = BeautifulSoup('<html><head><title>Your Title Here</title></head><body></body></html>', 'html.parser')

    # Find the <body> tag within the HTML structure
    body_tag = soup.body

    # Create an h1 tag and add it to the body of the HTML document
    h1_tag = soup.new_tag('h1')
    h1_tag.string = model_name
    body_tag.append(h1_tag)

    # add list of datasets represented
    datasets = data[0]['summary'].keys()
    
    h2_tag = soup.new_tag('h2') 
    h2_tag.string = 'Datasets represented:'
    body_tag.append(h2_tag)
    for dataset in datasets:
        p_tag = soup.new_tag('p')
        p_tag.string = dataset
        body_tag.append(p_tag)

    # Add the DataFrame content to the body
    body_tag.append(BeautifulSoup(html, 'html.parser'))
    html = str(soup)
    # save html
    with open(f'./data/summary_{model_name}.html', 'w') as f:
        f.write(html)
