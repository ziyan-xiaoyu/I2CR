# 有时候key运行到一半就没有额度了，这时候需要排除未回答的问题

import json

def remove_high_confidence_data(input_file, output_file):
    # 读取原始JSON文件
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 初始化新数据字典
    new_data = {}

    # 遍历原始数据
    for key, value in data.items():
        # 检查是否有 "confidence" == 1.0 的情况
        has_high_confidence = False
        if "mention_imgdesc_Azure" in value and "Objects" in value["mention_imgdesc_Azure"]:
            print(1)
            for caption_detail in value["mention_imgdesc_Azure"]["Objects"]:
                print(caption_detail)
                print("\n")
                if caption_detail.get("confidence") == 1.0:
                    print(2)
                    has_high_confidence = True
                    break

        # 如果没有 "confidence" == 1.0，则保留该条数据
        if not has_high_confidence:
            new_data[key] = value

    # 将新数据写入输出文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(new_data, f, ensure_ascii=False, indent=4)
        print(f'挑选gpt回答过的数据，从{len(data)}条数据中挑选了{len(new_data)}条数据。')

# 输入文件和输出文件路径
input_file = '../dataset_wikidiverse/result/1024_test_100_1.json'
output_file = '../dataset_wikidiverse/result/1024_test_100_1_select.json'

# 调用函数
remove_high_confidence_data(input_file, output_file)