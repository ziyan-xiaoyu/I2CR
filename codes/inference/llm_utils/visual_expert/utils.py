import torch
import numpy as np
import copy
from PIL import ImageDraw, Image

def sortProbsRes(Probs_list):
    '''对候选实体相似性概率分数排序，返回idx的排序
    '''
    Probs = torch.FloatTensor(Probs_list)
    _, sorted_idx = Probs.topk(10, largest=True, sorted=True)
    return sorted_idx.tolist()

def getGTentityIdx(GTentity_dict, Candentity_list):
    '''获取GT实体在候选实体列表中的idx
    '''
    label = [1 if i['name']==GTentity_dict['name'] else 0 for i in Candentity_list]
    # index = [idx for idx, value in list(enumerate(label)) if value == 1]
    try:
        GT_indice = label.index(1)
    except ValueError:
        GT_indice = 'nil'
    return GT_indice

def evaluate_probs(dataset, probs_key):
    top1 = 0; top3 = 0; top5 = 0; top7 = 0; notnil_total = 0; all_total = 0
    top1_var = []; top3_var = []; top5_var = []; top7_var = []; notnil_var = []; nil_var = []; all_var = []
    for sample_id, sample in dataset.items():
        GT_indice = getGTentityIdx(sample['entity'], sample['Candentity'])
        sorted_probs_idx = sortProbsRes(sample[probs_key])
        all_total += 1
        all_var.append(np.var(sample[probs_key]))
        if GT_indice != 'nil':
            notnil_total += 1
            notnil_var.append(np.var(sample[probs_key]))
            if GT_indice in sorted_probs_idx[:1]:
                top1 += 1
                top1_var.append(np.var(sample[probs_key]))
            if GT_indice in sorted_probs_idx[:3]:
                top3 += 1
                top3_var.append(np.var(sample[probs_key]))
            if GT_indice in sorted_probs_idx[:5]:
                top5 += 1
                top5_var.append(np.var(sample[probs_key]))
            if GT_indice in sorted_probs_idx[:7]:
                top7 += 1
                top7_var.append(np.var(sample[probs_key]))
        else:
            nil_var.append(np.var(sample[probs_key]))
    print('GT实体非nil结果  全体结果  方差')
    print('top1:--{}--{}--{}'.format(top1/notnil_total, top1/all_total, np.mean(top1_var)))
    print('top3:--{}--{}--{}'.format(top3/notnil_total, top3/all_total, np.mean(top3_var)))
    print('top5:--{}--{}--{}'.format(top5/notnil_total, top5/all_total, np.mean(top5_var)))
    print('top7:--{}--{}--{}'.format(top7/notnil_total, top7/all_total, np.mean(top7_var)))
    print('总体方差：{}; nil样本的方差：{}; 非nil样本的方差：{}'.format(np.mean(all_var), np.mean(nil_var), np.mean(notnil_var)))
    return

class CropImg(object):
    def __init__(self, image):
        self.image = image
    
    def _xywh2lurl(self, xywh_list):
        x = xywh_list[0]
        y = xywh_list[1]
        w = xywh_list[2]
        h = xywh_list[3]
        return (x, y, x + w, y + h)

    def crop_lurl(self, bbox_lurl):
        return self.image.crop(bbox_lurl)

    def crop_xywh(self, bbox_xywh):
        return self.crop_lurl(self._xywh2lurl(bbox_xywh))
    
    def showbbox_lurl(self, *bbox_lurl):
        color = [(0,0,0),(255,255,0),(25,25,112),(255,0,0),(0,0,255)]
        width=1
        image = copy.deepcopy(self.image)
        draw = ImageDraw.Draw(image)

        i = 0
        for bbox in bbox_lurl:
            draw.rectangle(bbox, outline=color[i%len(color)], width=width)
            i += 1
        return image
    
    def showbbox_xywh(self, *bbox_xywh):
        bbox_lurl = [self._xywh2lurl(bbox) for bbox in bbox_xywh]
        return self.showbbox_lurl(*bbox_lurl)

