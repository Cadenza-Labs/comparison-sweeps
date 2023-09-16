import json
import os
from rich import print
import pandas as pd
# get all summaries by getting files that are summary_*.json
model_summaries = {}
for file in os.listdir('./data'):
    if file.startswith('summary_') and file.endswith('.json') and '7b' not in file:
        with open(f'./data/{file}', 'r') as f:
            name = file.split('.')[0]
            # without the summary_
            name = name[8:]
            model_summaries[name] = json.load(f)
def hashconfig(config):
    return frozenset(config.items())

df_all = pd.DataFrame()
for model, summaries in model_summaries.items():
    hashedconfigs = [hashconfig(summary['parsed_config']) for summary in summaries]
    for summary in summaries:
        for ds, values in summary['summary'].items():
            values['ds'] = ds
            df = pd.DataFrame(values, index=[0])
            for config_name, val in summary['parsed_config'].items():
                df[config_name] = val
            df_all = pd.concat([df_all, df])
print(df_all)
# save
df_all.to_csv('./data/summary_all.csv')

# every summary has a parsed config
# want to aggregate by config and make a new summary, take mean over each of the values inside summmary 

