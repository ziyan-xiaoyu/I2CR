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
{'mention': 'Governor Ed Rendell', 'mention context text': 'Governor Ed Rendell announced…', 'candidate entities': {'1. Don Rendell': '…', '2. Stephen Rendell March': '…', '3. Ed Rendell': '…', '4. Stuart Rendell': '…', '5. Midge Rendell': '…', '6. Kenneth W. Rendell': '…', '7. Edward "Ed" G. Rendell': '…', '8. Ruth Rendell': '…', '9. Mike Rendell': '…', '10. Rendell': '…'}}

OUTPUT CoT:
Based on the given information, the mention is "Governor Ed Rendell" and the mention context text is… We need to choose one entity from ten candidates that best matches:
Step 1: Analyze the mention and context:
...
Step 2: Compare the mention with the candidate entities following these steps:
(1)Substitute each candidate entity into the original description text in turn to form a new text;
(2)Compare the new text with the description text of the candidate entity;
(3)Determine whether there is a semantic inconsistency between the two texts; if so, discard the candidate;
(4)Return to 1 and substitute the next candidate.
...
Step 3: Select the most relevant entity:
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
    ask_dict = {'mention name': 'Grzegorz Krychowiak',
                'mention context': 'Grzegorz Krychowiak on the right after signing the five-year contract agreement.',
                'candidate entities': {'1. Grzegorz Skrzecz': 'Grzegorz Skrzecz (25 August 1957 – 15 February 2023) was a Polish boxer, world championship medalist, actor, and the twin brother of Paweł Skrzecz.',
 '2. Wojciech Rychlik': 'Wojciech Rychlik is a biologist and photographer, born in Poland and living in the USA since 1980. Rychlik received his Ph.D. in 1980 from the Polish Academy of Sciences.',
 '3. Kazimierz Zdziechowski': 'Kazimierz Zdziechowski, also known under pseudonyms Władysław Zdora, Władysław Mouner, was a Polish landowner, prose writer, publicist, literary critic and novelist.',
 '4. Włodzimierz Korcz': 'Włodzimierz Korcz is a Polish composer, pianist, music producer and author of many popular songs. He graduated from the Academy of Music in Łódź and debuted in 1965.',
 '5. Grzegorz Krychowiak': 'Grzegorz Krychowiak ; born 29 January 1990) is a Polish professional footballer who plays for Lokomotiv Moscow and the Poland national team as a defensive midfielder.',
 '6. Grzegorz Przemyk': "Grzegorz Przemyk was an aspiring Polish poet from Warsaw, who was murdered by members of the Communist People's Milicja Obywatelska.",
 '7. Krzywcza': 'Krzywcza [ˈkʂɨft͡ʂa] is a village in Przemyśl County, Subcarpathian Voivodeship, in south-eastern Poland. It is the seat of the gmina called Gmina Krzywcza.',
 '8. Zbigniew Rychlicki': "Zbigniew Rychlicki was a Polish graphic artist, and illustrator of children's books. He received the Hans Christian Andersen Awards Prize given by the Jury of the International Board on Books for Young People for outstanding artistic achievement.",
 '9. Olivier Giroud': 'Olivier Jonathan Giroud is a French professional footballer who plays as a forward for Premier League club Chelsea and the France national team.',
 '10. Wojciech Jaruzelski': "Wojciech Witold Jaruzelski ; 6 July 1923 – 25 May 2014) was a Polish military officer, politician and de facto dictator of the Polish People's Republic from 1981 until 1989."}
                }
    return1, return2 = getPrompt(ask_dict)
    print(return1)
    print("###")
    print(return2)
    print("###")
