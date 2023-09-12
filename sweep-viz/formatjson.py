import json 

def get_json(path):
    with open(path, 'r') as f:
        return json.load(f)
    
def write_json(path):
    with open(path, 'w') as f:
        json.dump(get_json(path), f, indent=4)

get_json('sweep_to_summaries')