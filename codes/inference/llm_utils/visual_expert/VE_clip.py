import os
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

class CLIP4MEL(object):
    def __init__(self, device='cuda:2', clip_path='./CLIP/CLIP_ViT_bigG_14_laion2B_39B_b160k', mentionImg_root='./IMGs/wikidiverse_mentionIMG'):
        self.device = device
        self.model = CLIPModel.from_pretrained(clip_path).to(device)
        self.processor = CLIPProcessor.from_pretrained(clip_path)

        self.mentionImg_root = mentionImg_root

    def metaInfernece(self, image, text_list):
        with torch.no_grad():
            inputs = self.processor(text=text_list, images=image, return_tensors="pt", 
                                    max_length=77, padding='max_length', truncation=True).to(self.device)
            outputs = self.model(**inputs)
            logits_per_image = outputs.logits_per_image # image-text similarity score
            # probs = logits_per_image.softmax(dim=1) # take softmax to get probabilities
        return logits_per_image
    
    def contrast_ImgText(self, image, text_list):
        with torch.no_grad():
            logits_per_image = self.metaInfernece(image, text_list)
            probs = logits_per_image.softmax(dim=1) # take softmax to get probabilities
        return probs[0].tolist()
    
    def getCandsProbs(self, sample, desc_key='desc'):
        mention_imgpath = sample['mention_imgpath']
        try:
            mention_img = Image.open(os.path.join(self.mentionImg_root, mention_imgpath))
        except FileNotFoundError:
            print('mention图片无法打开，将返回默认结果')
            return [0.1]*10
        
        Candentity = sample['Candentity']
        cands_text_list = [i[desc_key] for i in Candentity]

        probs_list = self.contrast_ImgText(mention_img, cands_text_list)

        return probs_list

