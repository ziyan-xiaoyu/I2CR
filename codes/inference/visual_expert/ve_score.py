import os
import json
import re
from tqdm import tqdm
from PIL import Image
from llm_utils.utils import *
from llm_utils.visual_expert.VE_clip import CLIP4MEL

mention_imgroot = './IMGs/wikidiverse_mentionIMG'


Clip = CLIP4MEL(device='cuda:1', mentionImg_root=mention_imgroot)

def run_clip(dataset, save_path, num_cand4test):
    for sample_id, sample in tqdm(dataset.items()):
        mention_imgpath = sample['mention_imgpath']
        cand_entities = sample['Candentity'][:num_cand4test]
        if mention_imgpath != 'nil_img':
            mention_img = Image.open(os.path.join(mention_imgroot, mention_imgpath))
        else:
            mention_img = 'nil_img'
        
        if not isinstance(mention_img, str):
            cand_desc_list = [i['desc'] for i in cand_entities]
            cand_desc_list = [re.sub(u"\\(.*?\\)", "", d) for d in cand_desc_list]
            
            cand_score_list = Clip.metaInfernece(mention_img, cand_desc_list)
            cand_score_list = [i.item() for i in cand_score_list[0]]
            
            for idx, score in enumerate(cand_score_list):
                dataset[sample_id]['Candentity'][idx]['score'] = score
        else:
            for idx, cand in enumerate(cand_entities):
                dataset[sample_id]['Candentity'][idx]['score'] = 'break'

    dumpdata(dataset, save_path)
    return

if __name__ == '__main__':
    dataset = loaddata('./input/WikiDiverse_testset.json')
    save_path = './output/WikiDiverse_testset_score.json'
    run_clip(dataset, save_path, 10)
