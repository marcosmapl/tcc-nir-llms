import json

with open("ml_100k.json", 'r') as f:
    data_ml_100k = json.load(f)

print(len(data_ml_100k))