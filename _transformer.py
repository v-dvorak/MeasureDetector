import os
import json

def process_json_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    if len(data["system_measures"]) == 0:
        return
    elif "bottom" in data["system_measures"][0]:
        return
    
    # Process system_measures
    new_annots = [[] for i in range(5)]
    for i, name in enumerate(["system_measures", "stave_measures", "staves", "systems", "grand_staff"]):
        for measure in data[name]:
            new_measure = {
                "left": measure["left"],
                "top": measure["top"],
                "right": measure["left"] + measure["width"],
                "bottom": measure["top"] + measure["height"]
            }
            new_annots[i].append(new_measure)

    # Construct the new JSON structure
    new_data = {
        "width": data["width"],
        "height": data["height"],
        "system_measures": new_annots[0],
        "stave_measures": new_annots[1],  # Placeholder for stave_measures
        "staves": new_annots[2],  # Placeholder for staves
        "systems": new_annots[3],
        "grand_staff": new_annots[4]
    }

    # Save the processed JSON file
    with open(file_path, 'w') as file:
        json.dump(new_data, file, indent=4)

def process_directory(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".json"):
                file_path = os.path.join(root, file)
                print(f"Processing {file_path}")
                process_json_file(file_path)

if __name__ == "__main__":
    process_directory("/home/dvoravo/MeasureDetector/datasets/mpp")
