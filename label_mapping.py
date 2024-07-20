import json
import collections

def get_label_mapping(file):
    # Load the JSON data
    with open(file, 'r') as f:
        cat_to_name = json.load(f)

    cat_to_name = {int(k): v for k, v in cat_to_name.items()}
    return cat_to_name


