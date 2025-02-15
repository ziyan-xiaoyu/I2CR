import json

def calculate_accuracy(data_path_12, data_path, answer_path):
    with open(data_path_12, 'r', encoding='utf-8') as file0:
        data_12 = json.load(file0)
    with open(data_path, 'r', encoding='utf-8') as file1:
        data = json.load(file1)
    with open(answer_path, 'r', encoding='utf-8') as file2:
        answer = json.load(file2)
    
    total_count = 0
    step1_match_count = 0
    step2_match_count = 0
    all_match_count = 0
    step1_unmatched_ids = []
    step1_unmatched_answers = []
    step2_unmatched_ids = []
    step2_unmatched_answers = []
    all_unmatched_ids = []
    all_unmatched_answers = []
    step1_nil_num = 0
    step2_nil_num = 0
    all_nil_num = 0

    for item_id, item_data in data_12.items():
        mention_answer = answer.get(item_id)
        step1_ans = item_data.get("ans_1", "")
        step2_ans = item_data.get("ans_2", "")
        
        total_count += 1
        if step1_ans == mention_answer:
            step1_match_count += 1
        else:
            step1_unmatched_ids.append(item_id)
            step1_unmatched_answers.append(mention_answer)
            if mention_answer == 'nil':
                step1_nil_num += 1

        if step2_ans == mention_answer:
            step2_match_count += 1
        else:
            step2_unmatched_ids.append(item_id)
            step2_unmatched_answers.append(mention_answer)
            if mention_answer == 'nil':
                step2_nil_num += 1
    
    for item_id, item_data in data.items():
        mention_answer = answer.get(item_id)
        ans = item_data.get("GPTans", "")
        
        if ans == mention_answer:
            all_match_count += 1
        else:
            all_unmatched_ids.append(item_id)
            all_unmatched_answers.append(mention_answer)
            if mention_answer == 'nil':
                all_nil_num += 1

    accuracy_1 = step1_match_count / total_count if total_count > 0 else 0
    accuracy_2 = step2_match_count / total_count if total_count > 0 else 0
    accuracy_3 = all_match_count / total_count if total_count > 0 else 0

    print(len(data))
    print(f"Accuracy_1: {accuracy_1:.2%}")
    print("Unmatched IDs:", step1_unmatched_ids)
    print("Unmatched answers:", step1_unmatched_answers)
    print("step1_nil_num:", step1_nil_num)
    print('\n')
    print(f"Accuracy_2: {accuracy_2:.2%}")
    print("Unmatched IDs:", step2_unmatched_ids)
    print("Unmatched answers:", step2_unmatched_answers)
    print("step2_nil_num:", step2_nil_num)
    print('\n')
    print(f"Accuracy_3: {accuracy_3:.2%}")
    print("Unmatched IDs:", all_unmatched_ids)
    print("Unmatched answers:", all_unmatched_answers)
    print("all_nil_num:", all_nil_num)

data_path_12 = '../../../datasets/dataset_WikiMEL/result/step1_2_output.json'
data_path = '../../../datasets/dataset_WikiMEL/result/output.json'
answer_path = '../../../datasets/dataset_WikiMEL/WikiMEL_testset_label.json'

# data_path_12 = '../../../datasets/dataset_WikiDiverse/result/step1_2_output.json'
# data_path = '../../../datasets/dataset_WikiDiverse/result/output.json'
# answer_path = '../../../datasets/dataset_WikiDiverse/WikiDiverse_testset_label.json'

# data_path_12 = '../../../datasets/dataset_RichMEL/result/step1_2_output.json'
# data_path = '../../../datasets/dataset_RichMEL/result/output.json'
# answer_path = '../../../datasets/dataset_RichMEL/richpedia_testset_label.json'

calculate_accuracy(data_path_12, data_path, answer_path)

