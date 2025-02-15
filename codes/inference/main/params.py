# -*- coding: UTF-8 -*-

import os
import argparse

"""run which dataset"""
# __dataset_type = 'wikimel'
# __dataset_type = 'wikidiverse'
__dataset_type = 'richpedia'


"""num of parallel processes"""
# __num_cut = 10
__num_cut = 1


"""openai KEY"""
key = """"""

"""result save name"""
save_path = 'output.json'
step1_2_save_path = 'step1_2_output.json'

#######################################################
if __dataset_type == 'wikidiverse':
    __use_wikidiverse_bias_resnet = True
elif __dataset_type == 'wikimel' or __dataset_type == 'richpedia':
    __use_wikidiverse_bias_resnet = False

__num_cands = 10
_int2str = {10:'ten', 12:'twelve', 15:'fifteen', }
__num_cands_str = _int2str[__num_cands]

__cycle_num = 3

__wkeml_score_path = '../visual_expert/output/WikiMEL_testset_score.json'
__wkpd_score_path = '../visual_expert/output/WikiDiverse_testset_score.json'
__richpd_score_path = '../visual_expert/output/richpedia_testset_score.json'

if __dataset_type == 'wikimel':
    save_root = '../../../datasets/dataset_WikiMEL/result/'
elif __dataset_type == 'wikidiverse':
    save_root = '../../../datasets/dataset_WikiDiverse/result/'
else:
    save_root = '../../../datasets/dataset_RichMEL/result/'

__save_path = os.path.join(save_root, save_path)
__step1_2_save_path = os.path.join(save_root, step1_2_save_path)

__keys = """
"""


