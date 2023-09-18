import pandas as pd
import numpy as np
from rich import print
from dataclasses import dataclass
from typing import Callable, Tuple

@dataclass
class Experiment:
    partition_fn: Callable[[pd.DataFrame], Tuple[pd.DataFrame, pd.DataFrame]]
    names: (str, str)

def fn_a(df):
    # Condition 1: net=ccs, loss=1.0*ccs_prompt_var
    df_condition_1 = df[
        (df['net'] == 'ccs') & 
        (df['loss'] == '1.0*ccs_prompt_var') &
        (df['norm'] == 'leace') &
        (df['per probe prompt'] == False) &
        (df['erase_prompt'] == True)
    ]

    # Condition 2: net=ccs, loss=1.0*ccs
    df_condition_2 = df[
        (df['net'] == 'ccs') & 
        (df['loss'] == '1.0*ccs') &
        (df['norm'] == 'leace') &
        (df['per probe prompt'] == False) &
        (df['erase_prompt'] == True)
    ]
    return df_condition_1, df_condition_2

def fn_c(df):
    # Condition 1: net=ccs, norm=burns
    df_condition_1 = df[
        (df['net'] == 'ccs') & 
        (df['loss'] == '1.0*ccs') &
        (df['norm'] == 'burns') &
        (df['per probe prompt'] == False) &
        (df['erase_prompt'] == True)
    ]

    # Condition 2: net=ccs, norm=leace
    df_condition_2 = df[
        (df['net'] == 'ccs') & 
        (df['loss'] == '1.0*ccs') &
        (df['norm'] == 'leace') &
        (df['per probe prompt'] == False) &
        (df['erase_prompt'] == True)
    ]
    return df_condition_1, df_condition_2

def fn_d(df):
    # Condition 1: erase_prompt=True
    net = 'eigen'
    ppp = False
    ncw = 1.0
    df_condition_1 = df[
        (df['net'] == net) & 
        (df['loss'].isna()) &
        (df['norm'].isna()) &
        (df['neg_cov_weight'] == ncw) &
        (df['per probe prompt'] == ppp) &
        (df['erase_prompt'] == True)
    ]

    # Condition 2: erase_prompt=False
    df_condition_2 = df[
        (df['net'] == net) & 
        (df['loss'].isna()) &
        (df['norm'].isna()) &
        (df['neg_cov_weight'] == ncw) &
        (df['per probe prompt'] == ppp) &
        (df['erase_prompt'] == False)
    ]
    return df_condition_1, df_condition_2

def fn_e(df):
    # Condition 1: neg_cov_weight=1.0
    net = 'eigen'
    ppp = False
    erase_prompt = False
    df_condition_1 = df[
        (df['net'] == net) & 
        (df['loss'].isna()) &
        (df['norm'].isna()) &
        (df['neg_cov_weight'] == 1.0) &
        (df['per probe prompt'] == ppp) &
        (df['erase_prompt'] == erase_prompt)
    ]

    # Condition 2: neg_cov_weight=0.5
    df_condition_2 = df[
        (df['net'] == net) & 
        (df['loss'].isna()) &
        (df['norm'].isna()) &
        (df['neg_cov_weight'] == 0.5) &
        (df['per probe prompt'] == ppp) &
        (df['erase_prompt'] == erase_prompt)
    ]
    return df_condition_1, df_condition_2

exp_a = Experiment(fn_a, ('ccs_prompt_var', 'ccs'))
exp_c = Experiment(fn_c, ('ccs+burns', 'ccs+leace'))
exp_d = Experiment(fn_d, ('vinc erase prompt', 'vinc'))
exp_e = Experiment(fn_e, ('ncw=1.0', 'ncw=0.5'))
exps = [exp_a, exp_c, exp_d, exp_e]

def analyze_csv_results(csv_path, exp: Experiment):
    # Load the CSV
    df = pd.read_csv(csv_path)
    # breakpoint()
    # for col in df.columns:
    #     if all((s not in col for s in ["full", "partial", "none", 'sweep'])):
    #         print(f"unique values of {col}")
    #         print(df[col].unique())
    models = df['model'].unique()
    datasets = df['dataset'].unique()
    # Filter the dataframe for the two specified conditions
    cols_to_average = df.columns[-10:-1]

    df_condition_1, df_condition_2 = exp.partition_fn(df)

    filtermodel = lambda df, model: df[df['model'] == model]

    for model in models:
        means = []
        for df_cond in [df_condition_1, df_condition_2]:
            model_cond = filtermodel(df_cond, model)
            assert len(model_cond) == len(datasets)
            meaned = model_cond[cols_to_average].mean()
            means.append(meaned)
        # stack 1 and 2
        stacked = pd.concat(means, axis=1)
        print(f'[magenta]{model}[/magenta]')
        # rename cols
        stacked.columns = exp.names
        print(stacked)

    print(f'[magenta]Overall[/magenta]')
    stacked = pd.concat([df_condition_1[cols_to_average].mean(), df_condition_2[cols_to_average].mean()], axis=1)
    stacked.columns = exp.names
    print(stacked)


for exp in exps:
    analyze_csv_results('all_results.csv', exp)
    print("\n\n")
