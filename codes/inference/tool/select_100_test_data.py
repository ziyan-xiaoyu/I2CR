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

def sample_json_data(input_file, output_file, sample_size=100):
    """从 JSON 文件中随机选择指定数量的数据并保存"""
    data = load_json(input_file)
    total_records = len(data)

    if sample_size > total_records:
        raise ValueError(f"样本数量 {sample_size} 超过了数据集大小 {total_records}")

    # 随机选择 sample_size 条数据
    sampled_keys = random.sample(list(data.keys()), sample_size)
    sampled_data = {key: data[key] for key in sampled_keys}

    # 保存到新的 JSON 文件
    save_json(sampled_data, output_file)
    print(f"数据集一共{total_records} 条，成功保存 {sample_size} 条数据到 {output_file}")

if __name__ == "__main__":
    input_file = "../dataset_WIKIMEL/WikiMEL_testset.json"  # 输入 JSON 文件路径
    output_file = "../dataset_WIKIMEL/WikiMEL_testset_100.json"  # 输出 JSON 文件路径

    sample_json_data(input_file, output_file, sample_size=100)
