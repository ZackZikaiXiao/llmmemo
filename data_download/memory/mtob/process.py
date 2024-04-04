import json

# Define the path to the JSON file
file_path = 'llmmemo/data_download/memory/mtob/test_examples_ke.json'

# Read the JSON data from the file
with open(file_path, 'r', encoding='utf-8') as file:
    json_data = json.load(file)

# Process the JSON data
for item in json_data:
    if "translation" in item and item["translation"] == "--------------------":
        # Update 'translation' to the value of 'ground_truth' and remove 'ground_truth'
        item["translation"] = item.get("ground_truth", "")
        item.pop("ground_truth", None)

# Write the updated JSON data back to the file, overwriting the original
with open(file_path, 'w', encoding='utf-8') as file:
    json.dump(json_data, file, ensure_ascii=False, indent=4)

print("JSON file has been updated and overwritten.")
