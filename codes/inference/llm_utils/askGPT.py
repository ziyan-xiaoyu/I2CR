import time
import openai
import copy
# from openai.error import RateLimitError

import json
import requests
from typing import List

def send_request(url = None,
                 api_key = None,
                 model_name = "gpt-4o",
                 messages:List[dict] = None,
                 other_params = None,
    ):
    
    # https://platform.openai.com/docs/api-reference/chat
    default_other_params = {
        "temperature": 1,
        "top_p": 1,
        "frequency_penalty": 0,
        "presence_penalty": 0,
        "logprobs": False,
        "top_logprobs": None,
        "max_tokens": None,
        # "max_completion_tokens": None
    }
    default_other_params.update(other_params)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    payload = {
        "model": model_name,
        "messages": messages,
        **default_other_params
    }

    # print(json.dumps(payload, ensure_ascii=False, indent=2))
    response = requests.post(url, headers=headers, json=payload)
    
    if response.status_code == 200:
        to_return = response.json()
    else:
        raise Exception(f"{response.status_code}:\n{response.text}")

    return to_return

def askGPT_meta(messages, api_key):
    # url = "https://api.mnxcc.com/v1/chat/completions"
    url = "https://api.foureast.cn/v1/chat/completions"
    # 把这个key设置成list
    # api_key = 'sk-UcMYqh4gwhf7mwA9A9B6D249D30942E58d80316d4e89E226'
    api_key = api_key
    model_name="gpt-4o"
    other_params={
        "temperature": 0,
        "max_tokens": 4096
    }

    response = send_request(url=url, api_key=api_key, model_name=model_name, messages=messages, other_params=other_params)

    return response["choices"][0]["message"]["content"]


# def askGPT_meta(messages): 原来的

#     # openai.api_base = "https://api.foureast.cn/v1"
#     openai.api_base = "https://api.mnxcc.com/v1"
#     completion = openai.chat.completions.create(
#         # model="gpt-3.5-turbo-16k-0613",
#         model="gpt-4o",
#         messages=messages,
#         temperature=0,
#         # seed=42,
#     )
#     return completion.choices[0].message.content


class ASK_GPT(object):
    def __init__(self, key_list: list, time_sleep=0, error_sleep=20, num_retry=10):
        self.time_sleep = time_sleep
        self.error_sleep = error_sleep
        self.num_retry = num_retry

        openai.api_key = key_list[0]
        self.api_key = key_list[0]
        # sk-UcMYqh4gwhf7mwA9A9B6D249D30942E58d80316d4e89E226
        # sk-TNkjCHUPTUJYjvRs2eEf9a83930f49F0BdAa67224169A9A2
        # sk-gkshgbQa1DjkOvLi3147A623162d4106B44558C7E4A130B1
        # openai.api_key = 'sk-vW12ePHTYsKp7a2HE50644FeA36b4e599d9f14EfA932016f'
    
    def askGPT4Use_nround(self, messages):
        res = 'None'; status_code = 0

        for _ in range(self.num_retry):
            status_code += 1
            
            try:
                time.sleep(self.time_sleep)
                res = askGPT_meta(messages, self.api_key)
                status_code -= 1
                break
            # except RateLimitError as e:
            #     print(e)
            #     time.sleep(self.error_sleep)
            except Exception as e:
                print(e)
                time.sleep(self.error_sleep)

        if status_code == self.num_retry:
            print('#'*42 + '失败')
        return res

if __name__ == '__main__':
    key = ''
    ask_gpt = ASK_GPT([key], error_sleep=1, num_retry=2)

    messages = [
        {"role": "system", "content": "You are a rational data reviewer"},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Who won the world series in 2020?"},
                ],
        }
    ]

    print(ask_gpt.askGPT4Use_nround(messages=messages))


# if __name__ == "__main__": 原来的
#     messages=[
#         {"role": "system", "content": "你是一个nlp专家"}, 
#         {"role": "user", "content": "你是谁"}
#     ]

#     print(askGPT_meta(messages))