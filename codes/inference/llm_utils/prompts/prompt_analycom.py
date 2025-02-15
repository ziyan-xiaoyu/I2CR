# 从提及的文字描述中，提取关键的信息（这个暂时没用上）

import sys
sys.path.append('..')
import main.params as params
from llm_utils.utils import *


SYSTEM_INFO_INIT = '''You are a natural language processing expert.
I want you to perform entity linking task, linking ambiguous mention in text to its corresponding entity in knowledge base. 
I will provide a mention with its context text, as well as {num_cands} candidate entities with their descriptions. Based on all the available information, you need choose one entity that is most likely to correlate to the mention. 
After you provide an answer, please extract key points from the description text of the entity you selected, which can serve as supporting evidence to prove that the mention corresponds to the entity you selected.'''
SYSTEM_INFO = SYSTEM_INFO_INIT.format(num_cands=params.__num_cands_str)


USER_CONTENT_1 = 'GIVEN:\n{ask_dict_1}\nOUTPUT:\nThink step by step and provide the final answer.'

USER_CONTENT_2 = """Sure, the entity you have chosen is \"{entity1}\". Now, I will provide you with a more detailed description text of \"{entity1}\". Please extract key points from it, which can serve as supporting evidence to prove that the mention corresponds to the entity \"{entity1}\".
You need to follow these requirements: 
1. The key points you extract must be from the description text of \"{entity1}\" without adding any additional information that does not exist in the description text. 
2. Starting from these key points, it can be demonstrated that the mention corresponds to the entity \"{entity1}\".
3. Please provide the extracted key points directly without explanation or answering other content.

Here is a demo:
GIVEN:
...
OUTPUT:
Paralympic wheelchair rugby competitor, silver medal at 2008 Beijing Paralympics, gold medals at 2012 London and 2016 Rio Paralympics.

Now back to the current task. 
The more detailed description of \"{entity1}\" is as follows: {entity1_description}. Please extract the key points from it.
I hope your answer can be as concise as possible (Within 80 words limit), using phrases and keywords whenever possible.
Please provide the extracted key points directly without explanation or answering other content.
OUTPUT:
"""

TASK_RESNET = """"""


def getPrompt(ask_dict_1, GPTres, ask_dict_current, backboneRes_key=''):
    # ask_dict = {'name': '7. Wojciech Rychlik',
    #             'desc': '...'}
    if backboneRes_key == 'PTres':
        SystemInfo = SYSTEM_INFO
    elif backboneRes_key == 'Ires':
        SystemInfo = SYSTEM_INFO.replace('I will provide a mention with its context text,', 'I will provide a mention with its context text and image information,')

    user_content_1_new = USER_CONTENT_1.format(ask_dict_1=ask_dict_1)

    entity_name = ask_dict_current['name']
    entity_desc = ask_dict_current['desc']
    user_content_2_new = USER_CONTENT_2.format(entity1=entity_name, entity1_description={entity_name: entity_desc})

    return SystemInfo, user_content_1_new, GPTres, user_content_2_new

if __name__ == '__main__':
    ask_dict_1 = {'mention name': 'Grzegorz Krychowiak',
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
    GPTres = 'Grzegorz Przemyk was an aspiring Polish poet'
    ask_dict_current = {'name': '7. Wojciech Rychlik',
                        'desc': 'Turkey meat, commonly referred to as just turkey, is the meat from turkeys'}
    SystemInfo, user_content_1_new, GPTres, user_content_2_new = getPrompt(ask_dict_1, GPTres, ask_dict_current)
    print(SystemInfo)
    print('################################################################')
    print(user_content_1_new)
    print('################################################################')
    print(GPTres)
    print('################################################################')
    print(user_content_2_new)
