import json
import os
import openai
# from openai.error import RateLimitError
import multiprocessing
from tqdm import tqdm

import sys
sys.path.append('..')
from ex_ask_SFR import execute_dataset
# from ex_ask_SBert import execute_dataset
from llm_utils.askGPT import ASK_GPT
from llm_utils.utils import *
from llm_utils.infer_SFR import *
import params
import torch.multiprocessing as mp

# from gpu_detect import monitor_gpus
# gpus = ','.join(monitor_gpus())
# print(gpus)
# os.environ["CUDA_VISIBLE_DEVICES"] = gpus

dataset_type = params.__dataset_type

wkeml_score_path = params.__wkeml_score_path
wkpd_score_path = params.__wkpd_score_path
richpd_score_path = params.__richpd_score_path


def cutDataset(dataset, save_dir='../main/cut_temp/', num_cut=params.__num_cut):
    keys = list(dataset.keys())
    total_len = len(keys)
    truncation = np.linspace(0, total_len, num_cut+1)
    truncation = [int(t) for t in truncation]
    sub_dicts = []

    for i in range(num_cut):
        sub_dict = {k: dataset[k] for k in keys[truncation[i]:truncation[i+1]]}
        sub_dicts.append(sub_dict)
    
    for idx, sub_dict in enumerate(sub_dicts):
        dumpdata(sub_dict, os.path.join(save_dir, '{}.json'.format(idx+1)))
    return

def combine_res(num_cut=params.__num_cut):
    sup_dicts = []
    for i in range(num_cut):
        sup_dicts.append(loaddata('../main/cut_temp/{}_res.json'.format(i+1)))
    
    combine_dict = {}
    for sup_dict in sup_dicts:
        combine_dict.update(sup_dict)
    
    return combine_dict

def combine_step1_2_res(num_cut=params.__num_cut):
    sup_dicts = []
    for i in range(num_cut):
        sup_dicts.append(loaddata('../main/cut_temp/{}_step1_2_res.json'.format(i+1)))
    
    combine_dict = {}
    for sup_dict in sup_dicts:
        combine_dict.update(sup_dict)
    
    return combine_dict

class ModelLoader:
    '''加载两个模型：'''
    _llama_pipeline = None
    _SFR_tokenizer = None
    _SFR_model = None
    _SBert_model = None

    @staticmethod
    def get_llama_pipeline():
        if ModelLoader._llama_pipeline is None:
            print('load llama3')
            ModelLoader._llama_pipeline = load_model_llama(model_id='/data/yuguangya/ALLYOUNEED/7B/llama/chat/llama3-8b-instruct', max_length=4096)
        return ModelLoader._llama_pipeline

    @staticmethod
    def get_SFR_model():
        if ModelLoader._SFR_model is None:
            print('load SFR')
            ModelLoader._SFR_tokenizer, ModelLoader._SFR_model = load_model_SFR(model_dir='/root/nas/someModels/models/SFR-Embedding-Mistral-huggingface')
        return ModelLoader._SFR_tokenizer, ModelLoader._SFR_model
    
    @staticmethod
    def get_SBert_model():
        if ModelLoader._SBert_model is None:
            print('load SBert')
            ModelLoader._SBert_model = load_model_SBert()
        return ModelLoader._SBert_model


def process_func(ex_id:int, exdataset, key_list, score_dataset):
    '''子进程函数 改成全局'''
    llama_pipeline = ModelLoader.get_llama_pipeline()
    SFR_tokenizer, SFR_model = ModelLoader.get_SFR_model()
    # SBert_model = ModelLoader.get_SBert_model()

    save_path = '../main/cut_temp/{}_res.json'.format(ex_id+1)
    temp_path = '../main/cut_temp/{}_temp.json'.format(ex_id+1)
    step1_2_path = '../main/cut_temp/{}_step1_2_res.json'.format(ex_id+1)
    ask_gpt = ASK_GPT(key_list)

    is_record = True if ex_id == 1 else False
    execute_dataset(exdataset, save_path, temp_path, step1_2_path, ask_gpt, llama_pipeline, SFR_tokenizer, SFR_model, score_dataset, is_record)
    # execute_dataset(exdataset, save_path, temp_path, step1_2_path, ask_gpt, llama_pipeline, SBert_model, score_dataset, is_record)

    print('{} done'.format(ex_id+1))
    torch.cuda.empty_cache()  # 手动释放 CUDA 资源


class Multiprocess_exdataset(object):
    def __init__(self, keys:str, exdataset_list):

        keyList = keys.split('\n')
        
        self.num_possess = len(exdataset_list)
        # self.num_possess = 1
        every_key_list_len = int(len(keyList)/self.num_possess)
        self.keyListofList = [keyList[i:i + every_key_list_len] for i in range(0, len(keyList), every_key_list_len)]
        print('every key list len: ')
        for i in self.keyListofList:print(len(i))

        self.exdataset_list = exdataset_list

        if dataset_type == 'wikimel':
            self.score_dataset = loaddata(wkeml_score_path)
        elif dataset_type == 'wikidiverse':
            self.score_dataset = loaddata(wkpd_score_path)
        elif dataset_type == 'richpedia':
            self.score_dataset = loaddata(richpd_score_path)
        else: raise

    def err_call_back(self, err):
        print(f'error：{str(err)}')
    
    def flow(self):
        pool = multiprocessing.Pool(self.num_possess)

        for ex_id in range(self.num_possess):
            pool.apply_async(
                process_func,  # 直接使用全局函数
                args=(ex_id, self.exdataset_list[ex_id], self.keyListofList[ex_id], self.score_dataset),
                error_callback=self.err_call_back
            )

        pool.close()
        pool.join()


if __name__ == '__main__':
    import torch.multiprocessing as mp
    mp.set_start_method('spawn', force=True)  # 设置 spawn 模式
    print(params.__dataset_type)
    num_cut = params.__num_cut
    keys = params.__keys

    if params.__dataset_type == 'wikimel':
        dataset = loaddata('../dataset_WIKIMEL/WikiMEL_testset.json')
    elif params.__dataset_type == 'wikidiverse':
        dataset = loaddata('../dataset_wikidiverse/WikiDiverse_testset_summary.json')
    elif params.__dataset_type == 'richpedia':
        dataset = loaddata('../dataset_richpedia/richpedia_testset.json')    
    else:
        raise

    if len(dataset) > 2000:
        print(f'cut candidate entity to {params.__num_cands}')
        for sample_id, sample in dataset.items():
            sample['Candentity'] = sample['Candentity'][:params.__num_cands]
    
    print(len(dataset))
    cutDataset(dataset)

    for sample_id, sample in dataset.items():
        print(len(sample['Candentity']))
        break

    dataset_list = [loaddata('../main/cut_temp/{}.json'.format(i+1)) for i in range(num_cut)]
    print('every slice len：', len(dataset_list[0]))

    operate_test = Multiprocess_exdataset(keys=keys, exdataset_list=dataset_list)

    print('---start---')
    operate_test.flow()
    print('----------DONE----------')

    dumpdata(combine_res(num_cut=num_cut), params.__save_path)
    print(f'save to {params.__save_path}')
    dumpdata(combine_step1_2_res(num_cut=num_cut), params.__step1_2_save_path)
    print(f'save to {params.__step1_2_save_path}')


# 原始的 process_func
# def process_func(self, ex_id:int):
#     save_path = '../main/cut_temp/{}_res.json'.format(ex_id+1)
#     temp_path = '../main/cut_temp/{}_temp.json'.format(ex_id+1)
#     step1_2_path = '../main/cut_temp/{}_step1_2_res.json'.format(ex_id+1)
#     ask_gpt = ASK_GPT(self.keyListofList[ex_id])

#     is_record = True if ex_id == 1 else False

#     llama_pipeline = ModelLoader.get_llama_pipeline()
#     SFR_tokenizer, SFR_model = ModelLoader.get_SFR_model()

#     execute_dataset(self.exdataset_list[ex_id], save_path, temp_path, step1_2_path, ask_gpt, llama_pipeline, SFR_tokenizer, SFR_model, self.score_dataset, is_record)

#     print('{} done'.format(ex_id+1))
#     return

# 原始的 flow(self):
    # def flow(self):
    #     pool = multiprocessing.Pool(self.num_possess)

    #     for ex_id in range(self.num_possess):
    #         pool.apply_async(self.process_func, args=(ex_id, ), error_callback=self.err_call_back)

    #     pool.close()
    #     pool.join()