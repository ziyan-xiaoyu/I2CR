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

# def load_model_llama(model_id, max_length):
#     '''load llama3(no fine-tune）'''
#     # device = "cuda"
#     # model = Model.from_pretrained(
#     #     model_id,
#     #     device_map="auto",
#     #     max_length=max_length,
#     #     cache_dir='/root/nas/someModels/models/llama3_8B'
#     # )
    
#     device_map = {
#         f"model.layers.{i}": "cuda:0" if i < 16 else "cuda:1"  
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
#     '''top1 infer llama3(no fine-tune)'''
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
    '''load llama3(fine-tuned)'''
    model_dir = '../../../models/llama3-8b-instruct'
    ckpt_dir = '../../fine-tune/llama3/model_arg/llama3_8b_checkpoints'
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.float32,
        device_map='auto',
        # max_length=max_length
    )
    model = PeftModel.from_pretrained(model, ckpt_dir)    
    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        model_kwargs={"torch_dtype": torch.float32},
        # device_map=device_map,
        tokenizer=tokenizer
    )
    
    return pipeline


def llama3_infer(pipeline, messages):
    '''top1 infer llama(fine-tuned)'''
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
    '''entity descriptions summary'''
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


def load_model_SFR(model_dir):
    '''load SFR'''
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_auth_token=True)
    model = AutoModel.from_pretrained(
        model_dir, 
        use_auth_token=True,
        device_map='auto'
    )

    return tokenizer, model


def last_token_pool(last_hidden_states: torch.Tensor,
                    attention_mask: torch.Tensor,
                    normalize: bool = True) -> torch.Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        pooled_states = last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        pooled_states =  last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]
    
    if normalize:
        pooled_states = pooled_states / pooled_states.norm(p=2, dim=1, keepdim=True)
    
    return pooled_states

def SFR_calculate_score(tokenizer, model, mention, context, entity_name, entity_summary, mention_imgdesc_new, max_length, annlycom_th):
    
    # process mention & context
    mention_text = mention + context + mention_imgdesc_new + "."
    mention_batch = tokenizer(mention_text, max_length=max_length, padding=True, truncation=True, return_tensors="pt").to("cuda")
    mention_outputs = model(**mention_batch)
    mention_emb = last_token_pool(mention_outputs.last_hidden_state, mention_batch['attention_mask'], normalize=True)[0]
    
    # process entity & summary
    entity_text = entity + entity_summary
    entity_batch = tokenizer(entity_text, max_length=max_length, padding=True, truncation=True, return_tensors="pt").to("cuda")
    entity_outputs = model(**entity_batch)
    entity_emb = last_token_pool(entity_outputs.last_hidden_state, entity_batch['attention_mask'],  normalize=True)[0]
    
    # 计算相似性分数
    score = torch.dot(mention_emb, entity_emb).item()
    
    # 比较分数与阈值
    if score > annlycom_th:
        print(f'Reasonable, {score}')
        return 'Reasonable', score
    else:
        print(f'Unreasonable, {score}')
        return 'Unreasonable', score
