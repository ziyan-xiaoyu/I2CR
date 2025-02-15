# from functools import cache
import os
os.environ["CUDA_VISIBLE_DEVICES"]="3"
import torch
from tqdm import tqdm
import json,re
import torch.nn.functional as F
from torch import Tensor
import transformers
from PIL import Image,ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration, TextIteratorStreamer,AutoTokenizer, AutoModel, AutoModelForCausalLM
from peft import PeftModel
from modelscope import Model
from swift.tuners import Swift
from sentence_transformers import SentenceTransformer


'''
仿写调用gpt4的输入输出，做一个封装函数
输入：message，格式如下：
messages = [
        {'role':'system', 'content': SystemInfo},
        {'role':'user', 'content': user_content}
    ]
输出：res，格式如下：
response["choices"][0]["message"]["content"]
基本就是内容了
'''

'''
需要写的函数：
    对于llama3-8B
        1. 简化mention描述（这个比较短，没必要简写了）
        2. 简化entity描述（这个只对wikiDiverse）
        3. top10选1
        4. top10选3
    对于SFR-embed
        1. top10选1后的 1 pair 的相似度
        2. top10选3后的 3 pair 的相似度
'''

# 模型：llama3-8B，任务：top10选1，保存结果id和res(eg: 3. name)，不做评估
# 后期的评估函数，只需要sample的id和ans即可
# 模型：llama3-8B，任务：top10选3，保存结果id和res(eg: 3. name; ...)，不做评估
# 需要针对这个函数，再增加一个prompt文件，定制一个新的select_top3的prompt
# 简化entity描述(增加一个prompt文件)

# def load_model_llama(model_id, max_length):
#     '''加载模型 llama3(未微调）'''
#     # device = "cuda"
#     # model = Model.from_pretrained(
#     #     model_id,
#     #     device_map="auto",
#     #     max_length=max_length,
#     #     cache_dir='/root/nas/someModels/models/llama3_8B'
#     # )
    
#     device_map = {
#         f"model.layers.{i}": "cuda:0" if i < 16 else "cuda:1"  # 前 16 层到 GPU 0，后 16 层到 GPU 1
#         for i in range(32)
#     }
#     device_map["model.norm"] = "cuda:1"
#     device_map["lm_head"] = "cuda:1"
#     device_map["model.embed_tokens"] = "cuda:1"
#     model_dir = '/data/yuguangya/ALLYOUNEED/7B/llama/chat/llama3-8b-instruct'
#     # tokenizer = AutoTokenizer.from_pretrained(model_dir)
#     pipeline = transformers.pipeline(
#         "text-generation",
#         model=model_dir,
#         model_kwargs={"torch_dtype": torch.float32},
#         device_map=device_map,
#         # tokenizer=tokenizer
#     )
    
#     return pipeline


# def llama3_infer(pipeline, messages):
#     '''top1/top3推理 原模型(未微调)'''
#     prompt = pipeline.tokenizer.apply_chat_template(
#         messages, 
#         tokenize=False, 
#         add_generation_prompt=True
#     )
#     terminators = [
#         pipeline.tokenizer.eos_token_id,
#         pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
#     ]
#     try:
#         outputs = pipeline(
#             prompt,
#             max_new_tokens=256,
#             eos_token_id=terminators,
#             do_sample=True,
#             # 主要通过设置 temperature 和 top_p 来控制生成的多样性
#             temperature=0.9,
#             top_p=0.5,
#             pad_token_id=128001
#         )
#         response = outputs[0]["generated_text"][len(prompt):]
#         print(response)
#     except Exception as e:
#         print(e)
#         response = ''
#         print("infer error!")
    
#     return response


def load_model_llama(model_id, max_length):
    '''加载模型 llama3(微调之后)'''
    model_dir = '/data/yuguangya/ALLYOUNEED/7B/llama/chat/llama3-8b-instruct'
    ckpt_dir = '/root/nas/202409_SMCR/ARR_SMCR/finetuning/llama3/model_arg/20241213_1/checkpoint-1000'
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.float32,
        device_map='auto',
        # max_length=max_length
    )
    model = PeftModel.from_pretrained(model, ckpt_dir)    
    # device_map = {
    #     f"model.layers.{i}": "cuda:0" if i < 16 else "cuda:1"  # 前 16 层到 GPU 0，后 16 层到 GPU 1
    #     for i in range(32)
    # }
    # device_map["model.norm"] = "cuda:1"
    # device_map["lm_head"] = "cuda:1"
    # device_map["model.embed_tokens"] = "cuda:1"
    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        model_kwargs={"torch_dtype": torch.float32},
        # device_map=device_map,
        tokenizer=tokenizer
    )
    
    return pipeline


def llama3_infer(pipeline, messages):
    '''top1/top3推理 llama(微调之后的)'''
    prompt = pipeline.tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    terminators = [
        pipeline.tokenizer.eos_token_id,
        pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    try:
        outputs = pipeline(
            prompt,
            max_new_tokens=256,
            eos_token_id=terminators,
            do_sample=True,
            # 主要通过设置 temperature 和 top_p 来控制生成的多样性
            temperature=0.9,
            top_p=0.5,
            pad_token_id=128001
        )
        response = outputs[0]["generated_text"][len(prompt):]
        print(response)
    except Exception as e:
        print(e)
        response = ''
        print("infer error!")
    
    return response


def llama3_summary(pipeline, messages):
    '''为entity description写摘要'''
    prompt = pipeline.tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    terminators = [
        pipeline.tokenizer.eos_token_id,
        pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    try:
        outputs = pipeline(
            prompt,
            max_new_tokens=256,
            eos_token_id=terminators,
            do_sample=True,
            # 主要通过设置 temperature 和 top_p 来控制生成的多样性
            temperature=0.6,
            top_p=0.9,
            pad_token_id=128001
        )
        response = outputs[0]["generated_text"][len(prompt):]
        print(response)
    except Exception as e:
        print(e)
        response = ''
        print("summary error!")
    
    return response


# 调用SFR
'''
帮我改写上述代码，实现一下功能：
输入：
一个mention，以及这个mention对应的context
一个被选中的entity，以及这个entity对应的summary
处理：
对于mention和对应的context，text = context+ "\n" + name，然后做emb
对于被选中的entity和对应的summary，text = ent['name']+":"+ent['sum']，然后做emb
上面的这两个流程和源代码里的保持一致
最后只计算这两个emb之间的一个score，和一个阈值annlycom_th做一个比较，大于该阈值则返回'Reasonable'，小于则返回'Unreasonable'
'''
def load_model_SFR(model_dir):
    '''加载模型 SFR'''
    # device_map = {
    #     f"layers.{i}": "cuda:2" if i < 16 else "cuda:3"  # 前 16 层到 GPU 0，后 16 层到 GPU 1
    #     for i in range(32)
    # }
    # device_map["norm"] = "cuda:4"
    # device_map["embed_tokens"] = "cuda:4"
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_auth_token=True)
    model = AutoModel.from_pretrained(
        model_dir, 
        use_auth_token=True,
        # device_map=device_map
        device_map='auto'
    )

    return tokenizer, model


def last_token_pool(last_hidden_states: torch.Tensor,
                    attention_mask: torch.Tensor,
                    normalize: bool = True) -> torch.Tensor:
    """
    自定义池化函数，提取最后一个有效标记的隐藏状态，并进行可选的归一化处理。

    Args:
        last_hidden_states (torch.Tensor): 模型的最后隐藏状态，形状为 (batch_size, seq_len, hidden_dim)。
        attention_mask (torch.Tensor): 注意力掩码，形状为 (batch_size, seq_len)。
        normalize (bool): 是否对返回的嵌入向量进行归一化。默认为 True。

    Returns:
        torch.Tensor: 提取后的嵌入向量，形状为 (batch_size, hidden_dim)。
    """
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        pooled_states = last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        pooled_states =  last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]
    
    # 如果需要归一化，执行归一化操作
    if normalize:
        pooled_states = pooled_states / pooled_states.norm(p=2, dim=1, keepdim=True)
    
    return pooled_states

def SFR_calculate_score(tokenizer, model, mention, context, entity_name, entity_summary, mention_imgdesc_new, max_length, annlycom_th):
    """
    输入：mention 和 context，以及对应的选中 entity 和 summary。
    输出：mention 和 entity 的 embedding 相似性分数，并判断是否合理。
    """
    # import os
    # os.environ["CUDA_VISIBLE_DEVICES"]="4, 5"
    # device = torch.device("cuda:")
    # model = torch.nn.DataParallel(model).to(device)  # 自动将模型分布到多卡
    # model.to("cuda")
    
    # 处理 mention 和 context (再加上一个PTres类型，将视觉信息一并加进来)
    mention_text = context + mention_imgdesc_new + "."
    mention_batch = tokenizer(mention_text, max_length=max_length, padding=True, truncation=True, return_tensors="pt").to("cuda")
    mention_outputs = model(**mention_batch)
    mention_emb = last_token_pool(mention_outputs.last_hidden_state, mention_batch['attention_mask'], normalize=True)[0]
    
    # 处理 entity 和 summary
    entity_text = entity_summary
    entity_batch = tokenizer(entity_text, max_length=max_length, padding=True, truncation=True, return_tensors="pt").to("cuda")
    entity_outputs = model(**entity_batch)
    entity_emb = last_token_pool(entity_outputs.last_hidden_state, entity_batch['attention_mask'],  normalize=True)[0]
    
    # 计算相似性分数
    score = torch.dot(mention_emb, entity_emb).item()  # 点积表示相似性分数
    
    # 比较分数与阈值
    if score > annlycom_th:
        print(f'Reasonable, {score}')
        return 'Reasonable', score
    else:
        print(f'Unreasonable, {score}')
        return 'Unreasonable', score


def load_model_SBert():
    '''加载模型 SBert'''
    model = SentenceTransformer("/root/nas/someModels/models/SBert/all-MiniLM-L6-v2")

    return model


# 调用sentence-transformers
def SBert_STS_score(model, mention, context, entity_name, entity_summary, max_length, annlycom_th):
    """
    输入：mention 和 context，以及对应的选中 entity 和 summary。
    输出：mention 和 entity 的 语义相似度分数（默认余弦相似度），并判断是否合理。
    """

    # 处理 mention
    # mention_text = mention  + ":" + context
    mention_text = context
    embeddings1 = model.encode(mention_text)
    
    # 处理 entity  entity_summary
    # entity_text = entity_name + ":" + entity_summary
    entity_text = entity_summary
    embeddings2 = model.encode(entity_text)
    
    # 计算相似性分数（默认是余弦相似度）, model.similarity输出是tensor类型
    similarities = model.similarity(embeddings1, embeddings2)[0][0].item()
    
    # 比较分数与阈值
    if similarities > annlycom_th:
        print(f'Reasonable, {similarities}')
        return 'Reasonable', similarities
    else:
        print(f'Unreasonable, {similarities}')
        return 'Unreasonable', similarities
        
