import pandas as pd
from rich import print

def analyze_csv_results(csv_path):
    # Load the CSV
    df = pd.read_csv(csv_path)
    models = df['model'].unique()
    datasets = df['dataset'].unique()
    # Filter the dataframe for the two specified conditions
    cols_to_average = df.columns[-10:-1]

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

    filtermodel = lambda df, model: df[df['model'] == model]

    for model in models:
        means = []
        for df_cond in [df_condition_1, df_condition_2]:
            model_cond = filtermodel(df_cond, model)
            assert len(model_cond) == len(datasets)
            meaned = model_cond[cols_to_average].mean()
            means.append(meaned)
        stacked = pd.concat(means, axis=1)
        print(f'[magenta]{model}[/magenta]')
        print(stacked)

    print('overall')
    print(df_condition_1[cols_to_average].mean())
    print(df_condition_2[cols_to_average].mean())


analyze_csv_results('all_results.csv')