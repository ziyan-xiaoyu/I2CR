import os
# import torch
import json
import time
# from PIL import Image
from tqdm import tqdm
import sys
sys.path.append('')
from llm_utils.visual_expert.VE_azure import ex_dataset

def loaddata(path_json):
    with open(path_json, 'r', encoding='utf-8') as f: return json.load(f)

def dumpdata(obj, path_json):
    with open(path_json, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=4)
    return

if __name__ == '__main__':
    dataset = loaddata('')
    save_path = ''
    temp_path = ''

    ex_dataset(dataset, save_path, temp_path)

