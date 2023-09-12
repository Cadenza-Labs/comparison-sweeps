import json

def get_json(path):
    with open(path, 'r') as f:
        return json.load(f)
    
obj = get_json('sweep_to_summaries.json')

for sweep in obj:
    summary = sweep['summary']
    for dataset in summary:
        vals = summary[dataset].values()
        print(vals)
        total += sum(vals)
