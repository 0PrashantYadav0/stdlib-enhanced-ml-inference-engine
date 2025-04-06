import pandas as pd

# Load the CSV file
csv_path = "./Iris.csv"
df = pd.read_csv(csv_path)

features = df.iloc[:, 1:-1].values.tolist()
labels = df["Species"].astype("category").cat.codes.tolist() 

# Prepare JSON structure
json_data = {
    "data": features,
    "target": labels
}

json_data.keys(), len(json_data["data"]), len(json_data["target"])

# export json_data to a file
import json
with open("iris_data.json", "w") as json_file:
    json.dump(json_data, json_file)
# Load the JSON file to verify
with open("iris_data.json", "r") as json_file:
    loaded_json_data = json.load(json_file)
# Check the loaded data 