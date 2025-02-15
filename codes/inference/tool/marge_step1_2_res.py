import json
import os

def load_and_combine_json_files(folder_path, output_path):
    combined_data = {}

    for i in range(1, 11):
        file_name = f"{i}_step1_2_res.json"
        file_path = os.path.join(folder_path, file_name)
        
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)

                for entry in data:
                    sample_id = entry["sample_id"]
                    combined_data[sample_id] = entry

    with open(output_path, 'w', encoding='utf-8') as outfile:
        json.dump(combined_data, outfile, ensure_ascii=False, indent=4)
        print(f'total: {len(combined_data)}')


folder_path = '../main/cut_temp'
output_path = '../dataset_WIKIMEL/result/1112_step1_2_res_2.json'
load_and_combine_json_files(folder_path, output_path)

