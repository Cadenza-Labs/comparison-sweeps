from get_summary_json import get_summary
from render_summary_json import render_summary_json
import os
from pathlib import Path
import json
import yaml

if __name__ == '__main__':
    # get files in ../data
    import os
    from pathlib import Path

    data_directory = Path('./data').resolve()  # Replace with the correct path to your data directory
    all_sweep_dirs = [
        (data_directory / sweeps).resolve() 
        for sweeps in os.listdir(data_directory) 
        if (path := Path(os.path.join(data_directory, sweeps))).is_dir() and 
        path.name.endswith('_no_reporters')
    ]
    for sweeps in all_sweep_dirs:
        summary = get_summary(sweeps)
        model_name = Path(sweeps).name
        with open(f'./data/summary_{model_name}.json', 'w') as f:
            json.dump(summary, f, indent=4)
        render_summary_json(summary, model_name)

    
    # each sweep is a different command and sweeps across models and datasets

    # but in mine each sweep is a differnt model and dataset
