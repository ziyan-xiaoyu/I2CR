import sys
sys.path.append('..')
import main.params as params
from llm_utils.utils import *

#
SYSTEM_INFO_INIT = '''Your task is to create matches between the mention and candidate entities to select the best-matched entity for the given mention.'''
SYSTEM_INFO = SYSTEM_INFO_INIT.format(num_cands=params.__num_cands_str)


USER_CONTENT_1 = 'GIVEN:\n{ask_dict_1}\nOUTPUT:\nThink step by step and provide the final answer.'

#
USER_CONTENT_2 = """
Now verify if the selected entity is the correct answer.
The "After Replacement" context is the text created by replacing the mention with the selected entity in the "Original" context. Choose "<|ASSESSMENT|>: reasonable/unreasonable" as the final answer. Ignore minor grammatical errors; focus on changes in meaning.

Here are two examples for clarity:

Example 1:
GIVEN:
`Original`: The Venezuelan delegation at the Maccabiah Games.
`After Replacement`: The Venezuelans delegation at the Maccabiah Games.
OUTPUT CoT:
Step1: Compare the original and replaced texts for consistency in semantics, coherence, and logic:
(Analysis process)...
Step2: Determine whether the selected entity is reasonable:
<|ASSESSMENT|>: Unreasonable .

Example 2:
GIVEN:
`Original`: Moneygall, the village which Obama's great grandfather reportedly comes from.
`After Replacement`: Moneygall, the village which Barack Obama great grandfather reportedly comes from.
OUTPUT CoT:
Step1: Compare the original and replaced texts for consistency in semantics, coherence, and logic:
(Analysis process)...
Step2: Determine whether the selected entity is reasonable:
<|ASSESSMENT|>: Reasonable .

Let's begin verifying your answer.
GIVEN:
`Original`: {mention_context_text_Original}
`After Replacement`: {mention_context_text_Replaced}
OUTPUT CoT:
"""


def getPrompt(sample, GPTres, ask_type=None, ve_type=None):

    mention_name, mention_context, mention_imgpath, mention_imgdesc, Cands = getSample(sample)

    if ask_type == 'PTres':
        ask_dict_backbone = {'mention': mention_name,
                             'mention context text': mention_context,
                             'candidate entities': Cands}
        SystemInfo = SYSTEM_INFO

    elif ask_type == 'Ires':
        if isinstance(mention_imgdesc, dict):
            if isinstance(ve_type, str):
                mention_imgdesc_new = mention_imgdesc[ve_type]
            elif isinstance(ve_type, list):
                mention_imgdesc_new = dict([(key, mention_imgdesc[key]) for key in ve_type])
            else:
                raise
        else:
            mention_imgdesc_new = mention_imgdesc

        ask_dict_backbone = {'mention': mention_name,
                             'mention context text': mention_context,
                             'mention image information':mention_imgdesc_new,
                             'candidate entities': Cands}
        SystemInfo = SYSTEM_INFO.replace('Given a mention, its context,', 'Given a mention, its context, its image information,')
    
    candName_list = [i['name'] for i in sample['Candentity']]
    ans = getGPTans(GPTres, candName_list)

    if ans in ['nil', '0', ]:
        return 'break', 'break', 'break', 'break'

    ans_idx = candName_list.index(ans)
    ans_desc = getAheadSentence(sample['Candentity'][ans_idx]['desc'], restrict=224)
    ans = ans.replace('_', ' ')

    mention_context_replaced = mention_context.replace(mention_name, ans)

    user_content_1_new = USER_CONTENT_1.format(ask_dict_1=ask_dict_backbone)
    user_content_2_new = USER_CONTENT_2.format(entity1=ans, entity1_desc=ans_desc, mention_context_text_Original=mention_context, mention_context_text_Replaced=mention_context_replaced)

    # return SystemInfo, user_content_1_new, GPTres, user_content_2_new
    return mention_name, mention_context, ans, ans_desc


def askGhatgpt_info_draginAssess(sample, GPTres, ask_type=None, ve_type=None):
    SystemInfo, user_content_1, GPTres_1, user_content_2 = getPrompt(sample, GPTres, ask_type=ask_type, ve_type=ve_type)
    # print(f'{SystemInfo}\n{user_content_1}\n{GPTres_1}\n{user_content_2}')
    return SystemInfo, user_content_1, GPTres_1, user_content_2


if __name__ == '__main__':
    sample = {
        "mention": "YouTube",
        "mention_context": "Screenshot of the message YouTube visitors in Turkey used to find",
        "entity": {
            "name": "YouTube",
            "desc": "For the channel owned by the staff of the video hosting service, see YouTube .",
            "imgpath": [
                "YouTube/0.jpg",
                "YouTube/1.jpg",
                "YouTube/3.jpg",
                "YouTube/4.jpg",
                "YouTube/6.jpg",
                "YouTube/7.jpg",
                "YouTube/8.jpg"
            ],
            "desc_summary": "YouTube is a video hosting service with a channel owned by its staff."
        },
        "Candentity": [
            {
                "name": "YouTube_(Google)",
                "desc": "YouTube is an American online video sharing and social media platform headquartered in San Bruno, California, United States. Accessible worldwide, it was launched on February 14, 2005, by Steve Chen, Chad Hurley, and Jawed Karim. It is owned by Google and is the second most visited website, after Google Search. YouTube has more than 2.5 billion monthly users, who collectively watch more than one billion hours of videos each day. As of May 2019, videos were being uploaded at a rate of more than 500 hours of content per minute.In October 2006, YouTube was bought by Google for $1.65 billion. Google's ownership of YouTube expanded the site's business model, expanding from generating revenue from advertisements alone to offering paid content such as movies and exclusive content produced by YouTube. It also offers YouTube Premium, a paid subscription option for watching content without ads. YouTube also approved creators to participate in Google's AdSense program, which seeks to generate more revenue for both parties. YouTube reported revenue of $29.2 billion in 2022. In 2021, YouTube's annual advertising revenue increased to $28.8 billion, an increase in revenue of 9 billion from the previous year.Since its purchase by Google, YouTube has expanded beyond the core website into mobile apps, network television, and the ability to link with other platforms. Video categories on YouTube include music videos, video clips, news, short films, feature films, songs, documentaries, movie trailers, teasers, live streams, vlogs, and more. Most content is generated by individuals, including collaborations between YouTubers and corporate sponsors. Established media corporations such as Disney, Paramount, NBCUniversal, and Warner Bros. Discovery have also created and expanded their corporate YouTube channels to advertise to a greater audience.\nYouTube has had unprecedented social impact, influencing popular culture, internet trends, and creating multimillionaire celebrities. Despite its growth and success, it has been widely criticized for allegedly facilitating the spread of misinformation and the sharing of  copyrighted content, routinely violating its users' privacy, enabling censorship, and endangering child safety and wellbeing, and for its guidelines and how they are implemented.",
                "imgpath": [
                    "YouTube_(Google)/1.jpg",
                    "YouTube_(Google)/4.jpg",
                    "YouTube_(Google)/5.jpg",
                    "YouTube_(Google)/6.jpg",
                    "YouTube_(Google)/7.jpg",
                    "YouTube_(Google)/9.jpg"
                ],
                "desc_summary": "YouTube is an American online video sharing and social media platform owned by Google. It was launched in 2005 and has over 2.5 billion monthly users. YouTube offers free content with advertisements and a paid subscription option called YouTube Premium. It generates revenue through advertisements and partnerships with creators. YouTube has expanded to mobile apps, network television, and collaborations with media corporations. It has had a significant social impact but has also faced criticism for privacy concerns and copyright infringement."
            },
            {
                "name": "Youtube_accounts",
                "desc": "YouTube is an American online video sharing and social media platform headquartered in San Bruno, California, United States. Accessible worldwide, it was launched on February 14, 2005, by Steve Chen, Chad Hurley, and Jawed Karim. It is owned by Google and is the second most visited website, after Google Search. YouTube has more than 2.5 billion monthly users, who collectively watch more than one billion hours of videos each day. As of May 2019, videos were being uploaded at a rate of more than 500 hours of content per minute.In October 2006, YouTube was bought by Google for $1.65 billion. Google's ownership of YouTube expanded the site's business model, expanding from generating revenue from advertisements alone to offering paid content such as movies and exclusive content produced by YouTube. It also offers YouTube Premium, a paid subscription option for watching content without ads. YouTube also approved creators to participate in Google's AdSense program, which seeks to generate more revenue for both parties. YouTube reported revenue of $29.2 billion in 2022. In 2021, YouTube's annual advertising revenue increased to $28.8 billion, an increase in revenue of 9 billion from the previous year.Since its purchase by Google, YouTube has expanded beyond the core website into mobile apps, network television, and the ability to link with other platforms. Video categories on YouTube include music videos, video clips, news, short films, feature films, songs, documentaries, movie trailers, teasers, live streams, vlogs, and more. Most content is generated by individuals, including collaborations between YouTubers and corporate sponsors. Established media corporations such as Disney, Paramount, NBCUniversal, and Warner Bros. Discovery have also created and expanded their corporate YouTube channels to advertise to a greater audience.\nYouTube has had unprecedented social impact, influencing popular culture, internet trends, and creating multimillionaire celebrities. Despite its growth and success, it has been widely criticized for allegedly facilitating the spread of misinformation and the sharing of  copyrighted content, routinely violating its users' privacy, enabling censorship, and endangering child safety and wellbeing, and for its guidelines and how they are implemented.",
                "imgpath": [
                    "Youtube_accounts/0.jpg",
                    "Youtube_accounts/1.jpg",
                    "Youtube_accounts/3.jpg",
                    "Youtube_accounts/4.jpg",
                    "Youtube_accounts/6.jpg",
                    "Youtube_accounts/9.jpg"
                ],
                "desc_summary": "YouTube is an American online video sharing and social media platform owned by Google. It was launched in 2005 and has over 2.5 billion monthly users. YouTube offers free content with advertisements and a paid subscription option called YouTube Premium. It generates revenue through advertisements and partnerships with creators. YouTube has expanded to mobile apps, network television, and collaborations with media corporations. It has had a significant social impact but has also faced criticism for privacy concerns and copyright infringement. \n\nKeywords: YouTube, online video sharing, social media platform, Google, 2.5 billion monthly users, advertisements, YouTube Premium, revenue, mobile apps, network television, media corporations, social impact, criticism, privacy concerns, copyright infringement."
            },
            {
                "name": "YouTube",
                "desc": "For the channel owned by the staff of the video hosting service, see YouTube .",
                "imgpath": [
                    "YouTube/0.jpg",
                    "YouTube/1.jpg",
                    "YouTube/3.jpg",
                    "YouTube/4.jpg",
                    "YouTube/6.jpg",
                    "YouTube/7.jpg",
                    "YouTube/8.jpg"
                ],
                "desc_summary": "YouTube is a video hosting service with a channel owned by its staff."
            },
            {
                "name": "Internet_censorship_in_India",
                "desc": "Internet censorship in India is done by both central and state governments. DNS filtering and educating service users in suggested usages is an active strategy and government policy to regulate and block access to Internet content on a large scale. Also measures for removing content at the request of content creators through court orders have become more common in recent years. Initiating a mass surveillance government project like Golden Shield Project is also an alternative discussed over the years by government bodies. Main article: Save Your Voice",
                "imgpath": [
                    "Internet_censorship_in_India/0.jpg",
                    "supplyimg_1/229e6f18de43b44ccd7671a71332c170.png"
                ],
                "desc_summary": "Internet censorship in India is carried out by both central and state governments. It involves DNS filtering, educating service users, and court-ordered content removal. The government has also considered implementing a mass surveillance project called the Golden Shield Project."
            },
            {
                "name": "YouTube_(video_sharing_company)",
                "desc": "YouTube is an American online video sharing and social media platform headquartered in San Bruno, California, United States. Accessible worldwide, it was launched on February 14, 2005, by Steve Chen, Chad Hurley, and Jawed Karim. It is owned by Google and is the second most visited website, after Google Search. YouTube has more than 2.5 billion monthly users, who collectively watch more than one billion hours of videos each day. As of May 2019, videos were being uploaded at a rate of more than 500 hours of content per minute.In October 2006, YouTube was bought by Google for $1.65 billion. Google's ownership of YouTube expanded the site's business model, expanding from generating revenue from advertisements alone to offering paid content such as movies and exclusive content produced by YouTube. It also offers YouTube Premium, a paid subscription option for watching content without ads. YouTube also approved creators to participate in Google's AdSense program, which seeks to generate more revenue for both parties. YouTube reported revenue of $29.2 billion in 2022. In 2021, YouTube's annual advertising revenue increased to $28.8 billion, an increase in revenue of 9 billion from the previous year.Since its purchase by Google, YouTube has expanded beyond the core website into mobile apps, network television, and the ability to link with other platforms. Video categories on YouTube include music videos, video clips, news, short films, feature films, songs, documentaries, movie trailers, teasers, live streams, vlogs, and more. Most content is generated by individuals, including collaborations between YouTubers and corporate sponsors. Established media corporations such as Disney, Paramount, NBCUniversal, and Warner Bros. Discovery have also created and expanded their corporate YouTube channels to advertise to a greater audience.\nYouTube has had unprecedented social impact, influencing popular culture, internet trends, and creating multimillionaire celebrities. Despite its growth and success, it has been widely criticized for allegedly facilitating the spread of misinformation and the sharing of  copyrighted content, routinely violating its users' privacy, enabling censorship, and endangering child safety and wellbeing, and for its guidelines and how they are implemented.",
                "imgpath": [
                    "YouTube_(video_sharing_company)/1.jpg",
                    "YouTube_(video_sharing_company)/2.jpg",
                    "YouTube_(video_sharing_company)/3.jpg",
                    "YouTube_(video_sharing_company)/6.jpg",
                    "YouTube_(video_sharing_company)/8.jpg",
                    "YouTube_(video_sharing_company)/9.jpg"
                ],
                "desc_summary": "YouTube is an American online video sharing and social media platform owned by Google. It was launched in 2005 and has over 2.5 billion monthly users. YouTube offers free content with advertisements and a paid subscription option called YouTube Premium. It generates revenue through advertisements and partnerships with creators. YouTube has expanded to mobile apps, network television, and collaborations with media corporations. It has had a significant social impact but has also faced criticism for privacy concerns and copyright infringement. \n\nKeywords: YouTube, online video sharing, social media platform, Google, 2.5 billion monthly users, advertisements, YouTube Premium, revenue, mobile apps, network television, media corporations, social impact, criticism, privacy concerns, copyright infringement."
            },
            {
                "name": "Youtube_Fame",
                "desc": "YouTube is an American online video sharing and social media platform headquartered in San Bruno, California, United States. Accessible worldwide, it was launched on February 14, 2005, by Steve Chen, Chad Hurley, and Jawed Karim. It is owned by Google and is the second most visited website, after Google Search. YouTube has more than 2.5 billion monthly users, who collectively watch more than one billion hours of videos each day. As of May 2019, videos were being uploaded at a rate of more than 500 hours of content per minute.In October 2006, YouTube was bought by Google for $1.65 billion. Google's ownership of YouTube expanded the site's business model, expanding from generating revenue from advertisements alone to offering paid content such as movies and exclusive content produced by YouTube. It also offers YouTube Premium, a paid subscription option for watching content without ads. YouTube also approved creators to participate in Google's AdSense program, which seeks to generate more revenue for both parties. YouTube reported revenue of $29.2 billion in 2022. In 2021, YouTube's annual advertising revenue increased to $28.8 billion, an increase in revenue of 9 billion from the previous year.Since its purchase by Google, YouTube has expanded beyond the core website into mobile apps, network television, and the ability to link with other platforms. Video categories on YouTube include music videos, video clips, news, short films, feature films, songs, documentaries, movie trailers, teasers, live streams, vlogs, and more. Most content is generated by individuals, including collaborations between YouTubers and corporate sponsors. Established media corporations such as Disney, Paramount, NBCUniversal, and Warner Bros. Discovery have also created and expanded their corporate YouTube channels to advertise to a greater audience.\nYouTube has had unprecedented social impact, influencing popular culture, internet trends, and creating multimillionaire celebrities. Despite its growth and success, it has been widely criticized for allegedly facilitating the spread of misinformation and the sharing of  copyrighted content, routinely violating its users' privacy, enabling censorship, and endangering child safety and wellbeing, and for its guidelines and how they are implemented.",
                "imgpath": [
                    "Youtube_Fame/0.jpg",
                    "Youtube_Fame/1.jpg",
                    "Youtube_Fame/3.jpg",
                    "Youtube_Fame/5.jpg",
                    "Youtube_Fame/7.jpg",
                    "Youtube_Fame/9.jpg"
                ],
                "desc_summary": "YouTube is an American online video sharing and social media platform owned by Google. It was launched in 2005 and has over 2.5 billion monthly users. YouTube offers free content with advertisements and a paid subscription option called YouTube Premium. It generates revenue through advertisements and partnerships with creators. YouTube has expanded to mobile apps, network television, and collaborations with media corporations. It has had a significant social impact but has also faced criticism for privacy concerns and copyright infringement. \n\nKeywords: YouTube, online video sharing, social media platform, Google, 2.5 billion monthly users, advertisements, YouTube Premium, revenue, mobile apps, network television, media corporations, social impact, criticism, privacy concerns, copyright infringement."
            },
            {
                "name": "YOU_TUBE",
                "desc": "YouTube is an American online video sharing and social media platform headquartered in San Bruno, California, United States. Accessible worldwide, it was launched on February 14, 2005, by Steve Chen, Chad Hurley, and Jawed Karim. It is owned by Google and is the second most visited website, after Google Search. YouTube has more than 2.5 billion monthly users, who collectively watch more than one billion hours of videos each day. As of May 2019, videos were being uploaded at a rate of more than 500 hours of content per minute.In October 2006, YouTube was bought by Google for $1.65 billion. Google's ownership of YouTube expanded the site's business model, expanding from generating revenue from advertisements alone to offering paid content such as movies and exclusive content produced by YouTube. It also offers YouTube Premium, a paid subscription option for watching content without ads. YouTube also approved creators to participate in Google's AdSense program, which seeks to generate more revenue for both parties. YouTube reported revenue of $29.2 billion in 2022. In 2021, YouTube's annual advertising revenue increased to $28.8 billion, an increase in revenue of 9 billion from the previous year.Since its purchase by Google, YouTube has expanded beyond the core website into mobile apps, network television, and the ability to link with other platforms. Video categories on YouTube include music videos, video clips, news, short films, feature films, songs, documentaries, movie trailers, teasers, live streams, vlogs, and more. Most content is generated by individuals, including collaborations between YouTubers and corporate sponsors. Established media corporations such as Disney, Paramount, NBCUniversal, and Warner Bros. Discovery have also created and expanded their corporate YouTube channels to advertise to a greater audience.\nYouTube has had unprecedented social impact, influencing popular culture, internet trends, and creating multimillionaire celebrities. Despite its growth and success, it has been widely criticized for allegedly facilitating the spread of misinformation and the sharing of  copyrighted content, routinely violating its users' privacy, enabling censorship, and endangering child safety and wellbeing, and for its guidelines and how they are implemented.",
                "imgpath": [
                    "YOU_TUBE/2.jpg",
                    "YOU_TUBE/3.jpg",
                    "YOU_TUBE/4.jpg",
                    "YOU_TUBE/5.jpg",
                    "YOU_TUBE/6.jpg",
                    "YOU_TUBE/9.jpg"
                ],
                "desc_summary": "YouTube is an American online video sharing and social media platform owned by Google. It was launched in 2005 and has over 2.5 billion monthly users. YouTube offers free content with advertisements and a paid subscription option called YouTube Premium. It generates revenue through advertisements and partnerships with creators. YouTube has expanded to mobile apps, network television, and collaborations with media corporations. It has had a significant social impact but has also faced criticism for privacy concerns and copyright infringement."
            },
            {
                "name": "Virtual_YouTuber",
                "desc": "A virtual YouTuber or VTuber is an online entertainer who uses a virtual avatar generated using computer graphics. A growing trend that originated in Japan in the mid-2010s, a majority of VTubers are Japanese-speaking YouTubers or live streamers who use anime-inspired avatar designs. By 2020, there were more than 10,000 active VTubers. The first entertainer to use the phrase \"virtual YouTuber\", Kizuna AI, began creating content on YouTube in late 2016, and her popularity soon sparked a VTuber trend in Japan, and the establishment of other specialized agencies to promote them, such as Hololive Production, Nijisanji and VShojo. A rising number of fan translations and foreign-language VTubers have marked an increase in the phenomenon's international popularity. Virtual YouTubers have appeared in domestic advertising campaigns in Japan, and have broken live-stream-related world records.",
                "imgpath": [
                    "Virtual_YouTuber/0.jpg",
                    "Virtual_YouTuber/1.jpg",
                    "Virtual_YouTuber/2.jpg",
                    "Virtual_YouTuber/3.jpg"
                ],
                "desc_summary": "A virtual YouTuber (VTuber) is an online entertainer who uses a computer-generated avatar. Originating in Japan, VTubers are predominantly Japanese-speaking YouTubers or live streamers with anime-inspired avatars. There are over 10,000 active VTubers as of 2020. Kizuna AI, the first virtual YouTuber, popularized the trend in late 2016. VTubers have gained international popularity, with fan translations and foreign-language VTubers emerging. They have also appeared in Japanese advertising campaigns and set live-stream-related world records."
            },
            {
                "name": "YouTube_(channel)",
                "desc": "YouTube is YouTube's official YouTube channel for spotlighting videos and events on the platform. Events shown on the channel include YouTube Comedy Week and the YouTube Music Awards. Additionally, the channel uploads annual installments of YouTube Rewind. For a brief period in late 2013, the channel was ranked as the most-subscribed on the platform. As of April 2020, the channel has earned 31 million subscribers and 2 billion video views. Main articles: YouTube Rewind, its 2018 installment, and its 2019 installment",
                "imgpath": [
                    "supplyimg_1/66a88975f1ec47bf23ed8ba2c2576b03.png"
                ],
                "desc_summary": "YouTube's official channel for showcasing videos and events on the platform. It features YouTube Comedy Week, YouTube Music Awards, and annual YouTube Rewind installments. The channel was briefly the most-subscribed on YouTube in late 2013. It currently has 31 million subscribers and 2 billion video views."
            },
            {
                "name": "Youtube_Partner",
                "desc": "For the channel owned by the staff of the video hosting service, see YouTube .",
                "imgpath": [
                    "Youtube_Partner/0.jpg",
                    "Youtube_Partner/1.jpg",
                    "Youtube_Partner/4.jpg",
                    "Youtube_Partner/5.jpg",
                    "Youtube_Partner/6.jpg",
                    "Youtube_Partner/7.jpg",
                    "Youtube_Partner/8.jpg"
                ],
                "desc_summary": "YouTube is a video hosting service with a channel owned by its staff."
            }
        ],
        "label": {
            "name": "YouTube",
            "desc": "For the channel owned by the staff of the video hosting service, see YouTube ."
        },
        "mention_imgpath": "0c2265a7c21862f60bb2164d6e7869e1.jpg",
        "mention_imgdesc_Azure": {
            "Caption": "a computer screen with a message",
            "Dense Captions": "a screenshot of a computer; a computer screen shot of a message; a screenshot of a computer; a white background with red text",
            "Tags": "text; electronics; screenshot; software; web page; website; computer icon; operating system; computer; online advertising; office application software; multimedia software",
            "OCR text": "Bu siteye erişim mahkeme kararıyla; engellenmiştir.; T.C. Ankara 12. Sulh Ceza Mahkemesi 17:01/2008 Tanh ve 2008/55 nolu karan gereği be sbeye erişim engellenmiştir; Access to this web she has been suspended in accordance with decision no 2008/55 ol TR. Ankara 120 Cominal Court of Peace,",
            "Dense Captions detail": [
                {
                    "bounding_box": [
                        0,
                        0,
                        380,
                        285
                    ],
                    "caption": "a screenshot of a computer",
                    "Confidence": 1.0
                },
                {
                    "bounding_box": [
                        0,
                        0,
                        374,
                        279
                    ],
                    "caption": "a computer screen shot of a message",
                    "Confidence": 1.0
                },
                {
                    "bounding_box": [
                        0,
                        0,
                        376,
                        50
                    ],
                    "caption": "a screenshot of a computer",
                    "Confidence": 1.0
                },
                {
                    "bounding_box": [
                        12,
                        94,
                        339,
                        96
                    ],
                    "caption": "a white background with red text",
                    "Confidence": 1.0
                }
            ],
            "Objects": []
        },
        "mention_class": "3. Organization",
        "PTres": {
            "backbone": "Based on the given information, the mention is \"YouTube\" and the mention context text is \"Screenshot of the message YouTube visitors in Turkey used to find\". We need to select one entity from the ten candidate entities that is most likely to correlate with the mention. Let's go through the steps to determine the correct entity:\n\nStep 1: Analyze the mention and context:\nThe mention is \"YouTube\", which is a popular online video sharing and social media platform. The mention context text mentions visitors in Turkey using YouTube to find something.\n\nStep 2: Compare the mention with the candidate entities:\nLet's go through each candidate entity with its description and determine if they have any association with the mention:\n\n1. YouTube (Google): This candidate entity is associated with Google, but it doesn't provide any additional information that suggests it is the correct entity for the mention.\n\n2. Youtube accounts: This candidate entity is similar to the mention but doesn't provide any additional information that suggests it is the correct entity for the mention.\n\n3. YouTube: This candidate entity is the exact match for the mention and provides a description that aligns with the mention being an online video sharing and social media platform. This is a strong candidate.\n\n4. Internet censorship in India: This candidate entity is not related to the mention of YouTube.\n\n5. YouTube (video sharing company): This candidate entity is similar to the mention but doesn't provide any additional information that suggests it is the correct entity for the mention.\n\n6. Youtube Fame: This candidate entity is similar to the mention but doesn't provide any additional information that suggests it is the correct entity for the mention.\n\n7. YOU TUBE: This candidate entity is similar to the mention but doesn't provide any additional information that suggests it is the correct entity for the mention.\n\n8. Virtual YouTuber: This candidate entity is related to YouTubers who use virtual avatars, but it doesn't directly correspond to the mention of YouTube.\n\n9. YouTube (channel): This candidate entity is related to YouTube's official channel, but it doesn't directly correspond to the mention of YouTube.\n\n10. Youtube Partner: This candidate entity is related to the staff of the video hosting service, but it doesn't directly correspond to the mention of YouTube.\n\nStep 3: Select the most relevant candidate entities:\nBased on the analysis, the most relevant candidate entity is \"3. YouTube\" as it is the exact match for the mention and provides a description that aligns with the mention being an online video sharing and social media platform.\n\n<|ANSWER|>: 3. YouTube",
            "backbone_assess": "Based on the given information, the original sentence is \"Screenshot of the message YouTube visitors in Turkey used to find\", and the replaced sentence is \"Screenshot of the message YouTube visitors in Turkey used to find\".\n\nLet's analyze carefully to determine if the replaced sentence is reasonable, coherent, and semantically consistent with the original sentence:\n\nThe replaced sentence is the same as the original sentence, except for the mention \"YouTube\" being replaced with the entity \"YouTube\". Since the mention \"YouTube\" is replaced with the same entity, the replaced sentence remains logical and coherent. It maintains semantic consistency with the original sentence as the meaning and context of the sentence are preserved.\n\n<|ASSESSMENT|>: Reasonable",
            "assess": {
                "0": [
                    "break",
                    "break"
                ],
                "1": [
                    "break",
                    "break"
                ],
                "2": [
                    "break",
                    "break"
                ]
            },
            "annlycom": "channel owned by staff, video hosting service, see YouTube",
            "score": 36.520896911621094
        },
        "Ires": {
            "backbone": "Based on the given information, the mention is \"YouTube\", the mention context text is \"Screenshot of the message YouTube visitors in Turkey used to find\", and the image information describes \"a computer screen with a message\". We need to select one entity from the ten candidate entities that is most likely to correlate with the mention. Let's go through the steps to determine the correct entity:\n\nStep 1: Analyze the mention and context:\nThe mention \"YouTube\" is a popular online video sharing and social media platform. The mention context text mentions visitors in Turkey using YouTube to find something.\n\nStep 2: Analyze the mention image information and identify helpful details:\nThe image information describes a computer screen with a message, but it doesn't provide any specific details that can help in determining the correct entity.\n\nStep 3: Compare the mention with the candidate entities:\nLet's go through each candidate entity with its description and determine if they have any association with the mention:\n\n1. YouTube (Google): This entity is the same as the mention and describes the American online video sharing and social media platform.\n2. Youtube accounts: This entity is similar to the mention but doesn't provide any additional information.\n3. YouTube: This entity is similar to the mention and mentions a channel owned by the staff of the video hosting service.\n4. Internet censorship in India: This entity is not related to the mention and describes internet censorship in India.\n5. YouTube (video sharing company): This entity is the same as the mention and describes the American online video sharing and social media platform.\n6. Youtube Fame: This entity is similar to the mention but doesn't provide any additional information.\n7. YOU TUBE: This entity is similar to the mention but doesn't provide any additional information.\n8. Virtual YouTuber: This entity is not related to the mention and describes online entertainers who use virtual avatars.\n9. YouTube (channel): This entity is similar to the mention and describes YouTube's official channel for spotlighting videos and events.\n10. Youtube Partner: This entity is not related to the mention and describes a channel owned by the staff of the video hosting service.\n\nStep 4: Select the most relevant candidate entities:\nBased on the comparison, the most relevant candidate entities are:\n1. YouTube (Google)\n3. YouTube\n5. YouTube (video sharing company)\n9. YouTube (channel)\n\nAmong these options, the simplest name is \"YouTube\". Therefore, the answer is:\n\n<|ANSWER|>: 3. YouTube",
            "backbone_assess": "Based on the given information, the original sentence is \"Screenshot of the message YouTube visitors in Turkey used to find\", and the replaced sentence is \"Screenshot of the message YouTube visitors in Turkey used to find\".\n\nLet's analyze carefully to determine if the replaced sentence is reasonable, coherent, and semantically consistent with the original sentence:\n\nThe replaced sentence is exactly the same as the original sentence, with no changes made. Therefore, the replaced sentence is reasonable, coherent, and semantically consistent with the original sentence.\n\n<|ASSESSMENT|>: Reasonable",
            "assess": {
                "0": [
                    "break",
                    "break"
                ],
                "1": [
                    "break",
                    "break"
                ],
                "2": [
                    "break",
                    "break"
                ]
            },
            "annlycom": "channel owned by staff, video hosting service",
            "score": 36.520896911621094
        }
    }
    
    GPTres = '''Based on the given information, the mention is \"YouTube\" and the mention context text is \"Screenshot of the message YouTube visitors in Turkey used to find\". We need to select one entity from the ten candidate entities that is most likely to correlate with the mention. Let's go through the steps to determine the correct entity:\n\nStep 1: Analyze the mention and context:\nThe mention is \"YouTube\", which is a popular online video sharing and social media platform. The mention context text mentions visitors in Turkey using YouTube to find something.\n\nStep 2: Compare the mention with the candidate entities:\nLet's go through each candidate entity with its description and determine if they have any association with the mention:\n\n1. YouTube (Google): This candidate entity is associated with Google, but it doesn't provide any additional information that suggests it is the correct entity for the mention.\n\n2. Youtube accounts: This candidate entity is similar to the mention but doesn't provide any additional information that suggests it is the correct entity for the mention.\n\n3. YouTube: This candidate entity is the exact match for the mention and provides a description that aligns with the mention being an online video sharing and social media platform. This is a strong candidate.\n\n4. Internet censorship in India: This candidate entity is not related to the mention of YouTube.\n\n5. YouTube (video sharing company): This candidate entity is similar to the mention but doesn't provide any additional information that suggests it is the correct entity for the mention.\n\n6. Youtube Fame: This candidate entity is similar to the mention but doesn't provide any additional information that suggests it is the correct entity for the mention.\n\n7. YOU TUBE: This candidate entity is similar to the mention but doesn't provide any additional information that suggests it is the correct entity for the mention.\n\n8. Virtual YouTuber: This candidate entity is related to YouTubers who use virtual avatars, but it doesn't directly correspond to the mention of YouTube.\n\n9. YouTube (channel): This candidate entity is related to YouTube's official channel, but it doesn't directly correspond to the mention of YouTube.\n\n10. Youtube Partner: This candidate entity is related to the staff of the video hosting service, but it doesn't directly correspond to the mention of YouTube.\n\nStep 3: Select the most relevant candidate entities:\nBased on the analysis, the most relevant candidate entity is \"3. YouTube\" as it is the exact match for the mention and provides a description that aligns with the mention being an online video sharing and social media platform.\n\n<|ANSWER|>: 3. Youtube_Partner"'''

    SystemInfo, user_content_1, GPTres_1, user_content_2 = askGhatgpt_info_draginAssess(sample, GPTres, backboneRes_key='Ires')
    
    print(SystemInfo)
    print('######')
    print(user_content_1)
    print('######')
    print(GPTres_1)
    print('######')
    print(user_content_2)
    print('######')
