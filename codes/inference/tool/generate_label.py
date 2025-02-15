# 用于生成richpedia-mel的标答label.json

import json
import random

def load_json(file_path):
    """加载 JSON 文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(data, file_path):
    """保存数据为 JSON 文件"""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def generate_label(input_file, output_file):

    data = load_json(input_file)
    print(len(data))

    data_label = {}
    for key, sample in data.items():
        answer = sample.get('entity', {}).get('name', None)
        if answer:
            data_label[key] = answer
        else:
            data_label[key] = 'nil'
    save_json(data_label, output_file)
    print(len(data_label))

if __name__ == "__main__":
    input_file = "/root/nas/202409_SMCR/ARR_SMCR/SMCR_om_top1/dataset_richpedia/richpedia_testset.json"  # 输入 JSON 文件路径
    output_file = "/root/nas/202409_SMCR/ARR_SMCR/SMCR_om_top1/dataset_richpedia/richpedia_testset_label.json"  # 输出 JSON 文件路径

    generate_label(input_file, output_file)