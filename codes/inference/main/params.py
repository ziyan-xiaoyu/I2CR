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
save_path = '0120_test_cp_om_ft_top1_2.json'
step1_2_save_path = '0120_step1_2_om_ft_top1_2.json'

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
    save_root = '../dataset_WIKIMEL/result/'
elif __dataset_type == 'wikidiverse':
    save_root = '../dataset_wikidiverse/result/'
else:
    save_root = '../dataset_richpedia/result/'

__save_path = os.path.join(save_root, save_path)
__step1_2_save_path = os.path.join(save_root, step1_2_save_path)

__keys = """sk-4O1xvdln9enOjovV977bC0C3Ad79407796F5F196Eb5c4cD0
sk-a33nLrrD1S2gDsvvFdC64e7dDd144dA1B158Fa7e77999e8f
sk-Joo2ZBew2RlQKaUB54DbC5BfE0314a2380965921F3Bc22Df
sk-ilJ7T27o9eZkxvuR880bCc02F78c431bB39379603eF36fBb
sk-5zsHktIXL0ghDnAb9896Dc84D57a4110Ad46A124A599F314
sk-ZzGeIFpROlUQeWe7B211Ee15508b41B580317e42Df7d389e
sk-VRSqsCobmnYFxVFZ87F36cEcA9844976Ae253c4dB77a86A2
sk-Va5MvgeQCrukx2rjC545922b59244a409aAd84F3FcEb2699
sk-W95pFvVirQZrJEiO406053F9Fb3b459cB1FeEbC482F7A885
sk-gm7Ah9X50oWmBaH1E2D17a394f2d482f94Ea6a2716E47b5e
"""


# 9月份买的key(都用完了)
# sk-vW12ePHTYsKp7a2HE50644FeA36b4e599d9f14EfA932016f(用完了)
# sk-UcMYqh4gwhf7mwA9A9B6D249D30942E58d80316d4e89E226
# sk-TNkjCHUPTUJYjvRs2eEf9a83930f49F0BdAa67224169A9A2
# sk-gkshgbQa1DjkOvLi3147A623162d4106B44558C7E4A130B1

# 10月份买的key
# sk-4O1xvdln9enOjovV977bC0C3Ad79407796F5F196Eb5c4cD0
# sk-a33nLrrD1S2gDsvvFdC64e7dDd144dA1B158Fa7e77999e8f
# sk-Joo2ZBew2RlQKaUB54DbC5BfE0314a2380965921F3Bc22Df
# sk-ilJ7T27o9eZkxvuR880bCc02F78c431bB39379603eF36fBb
# sk-5zsHktIXL0ghDnAb9896Dc84D57a4110Ad46A124A599F314
# sk-ZzGeIFpROlUQeWe7B211Ee15508b41B580317e42Df7d389e
# sk-VRSqsCobmnYFxVFZ87F36cEcA9844976Ae253c4dB77a86A2
# sk-Va5MvgeQCrukx2rjC545922b59244a409aAd84F3FcEb2699
# sk-W95pFvVirQZrJEiO406053F9Fb3b459cB1FeEbC482F7A885
# sk-gm7Ah9X50oWmBaH1E2D17a394f2d482f94Ea6a2716E47b5e

