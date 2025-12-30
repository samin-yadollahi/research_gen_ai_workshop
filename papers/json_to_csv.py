
import json
import os
import pandas as pd

def flatten_json(y, parent_key='', sep='.'):
    items = []
    for k, v in y.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_json(v, new_key, sep=sep).items())
        elif isinstance(v, list):
            # Convert lists to string (CSV-friendly)
            items.append((new_key, str(v)))
        else:
            items.append((new_key, v))
    return dict(items)



json_dir = "papers"   # directory containing JSON files
records = []

for file_name in os.listdir(json_dir):
    if file_name.endswith(".json"):
        file_path = os.path.join(json_dir, file_name)

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        flat_data = flatten_json(data)
        flat_data["source_file"] = file_name  # traceability
        records.append(flat_data)

df = pd.DataFrame(records)
df.to_csv("output.csv", index=False, encoding="utf-8")
