import json

# Define your seed variable
seed_emo = "423"

# Create a dictionary with the seed variable
data = {
    'seed_emo': seed_emo
}

# Specify the file path where JSON data will be saved
json_file_path = 'seed_data.json'

# Write the data to a JSON file
with open(json_file_path, 'w') as json_file:
    json.dump(data, json_file)

print(f"JSON data saved to '{json_file_path}'")
