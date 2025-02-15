import sys
sys.path.append('..')
import main.params as params

#
SYSTEM_INFO_INIT = '''Your task is to create matches between the mention and candidate entities to select the best-matched entity for the given mention.'''
SYSTEM_INFO = SYSTEM_INFO_INIT.format(num_cands=params.__num_cands_str)

#
SYSTEM_INFO_ADD_1 = '''
If no candidate matches the mention, you ANSWER: nil.
'''
SYSTEM_INFO_ADD_2 = '''
The answer must be among the candidates, you cannot answer nil.
'''
#
TASK_COT_ICL = """GIVEN:
{'mention': 'Governor Ed Rendell', 'mention context text': 'Governor Ed Rendell announced…', 'mention image information': '…', 'candidate entities': {'1. Don Rendell': '…', '2. Stephen Rendell March': '…', '3. Ed Rendell': '…', '4. Stuart Rendell': '…', '5. Midge Rendell': '…', '6. Kenneth W. Rendell': '…', '7. Edward "Ed" G. Rendell': '…', '8. Ruth Rendell': '…', '9. Mike Rendell': '…', '10. Rendell': '…'}}

OUTPUT CoT:
Based on the given information, the mention is "Governor Ed Rendell", the mention context text is… and the image information indicates that… We need to choose one entity from ten candidates that best matches:
Step 1: Analyze the mention and context:
...
Step 2: Analyze the mention image information and identify helpful details:
...
Step 3: Compare the mention with the candidate entities:
Review each candidate entity and its description to identify any association with the mention, comparing them in detail.
...
Step 4: Select the most relevant entity:
<|ANSWER|>: 3. Ed Rendell
"""

#
TASK_RESNET = """
Choose only one option from candidate entities; no explanations or multiple choices.
Must use this format for the final answer: <|ANSWER|>: <The entity name you selected>. 
"""

#
TASK_RESNET_ADD_1 = """
Note: If no candidate matches the mention, you need ANSWER: nil."""

# TASK_RESNET_ADD_2 = """
# Among multiple candidate entities, the one with the simplest name is more likely to be the answer. For example, in [1. Wikipedia (website), 2. German Wikipedia, 3. Wikipedia], the answer is 3. Wikipedia. Of course, this rule doesn't always work."""

TASK_RESNET_ADD_2 = '''
Note: The answer must be among the candidates, you cannot answer nil.
'''


def getPrompt(ask_dict, dataset_type=params.__dataset_type, use_wikidiverse_bias=params.__use_wikidiverse_bias_resnet):
    assert dataset_type in ['wikidiverse', 'wikimel', 'richpedia']

    cand_list = [k for k,v in ask_dict['candidate entities'].items()]

    if dataset_type == 'wikidiverse':
        cand_list.append('nil')
        SystemInfo = SYSTEM_INFO + SYSTEM_INFO_ADD_1
        task_resnet_temp_1 = TASK_RESNET + TASK_RESNET_ADD_1
    elif dataset_type == 'wikimel' or dataset_type == 'richpedia':
        SystemInfo = SYSTEM_INFO
        task_resnet_temp_1 = TASK_RESNET
    else:
        raise

    if use_wikidiverse_bias == True:
        # task_resnet = task_resnet_temp_1 + TASK_RESNET_ADD_2
        task_resnet = task_resnet_temp_1
    else:
        task_resnet = task_resnet_temp_1

    task_resnet_new = task_resnet.format(CAND_LIST=cand_list)

    system_content = SystemInfo
    user_content = f'GIVEN:\n{ask_dict}\n{task_resnet_new}'
    return system_content, user_content

if __name__ == '__main__':
    ask_dict = {'mention name': 'a',
                'mention context': 'b',
                'candidate entities': {'1. a':'sdhf',
                                       '2. h': 'aus'},
                'mention image information': 'd'
                }
    return1, return2 = getPrompt(ask_dict)
    print(return1)
    print("###")
    print(return2)
    print("###")
