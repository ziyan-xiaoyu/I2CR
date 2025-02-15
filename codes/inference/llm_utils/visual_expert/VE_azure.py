import os
import json
import time
import azure.ai.vision as visionsdk
from tqdm import tqdm
from retrying import retry
from PIL import Image

VISION_ENDPOINT = 'https://ljw4mel.cognitiveservices.azure.com/'
VISION_KEY = '14ad3e7c2c75460797cb2f0fc41118a0'

service_options = visionsdk.VisionServiceOptions(VISION_ENDPOINT, VISION_KEY)
analysis_options = visionsdk.ImageAnalysisOptions()
analysis_options.features = (
    # visionsdk.ImageAnalysisFeature.CROP_SUGGESTIONS |
    visionsdk.ImageAnalysisFeature.CAPTION |
    visionsdk.ImageAnalysisFeature.DENSE_CAPTIONS |
    visionsdk.ImageAnalysisFeature.OBJECTS |
    visionsdk.ImageAnalysisFeature.TEXT |
    visionsdk.ImageAnalysisFeature.TAGS
    )
analysis_options.language = "en"
analysis_options.model_version = "latest"


def azure_analysisImg(img_path):
    vision_source = visionsdk.VisionSource(filename=img_path)
    image_analyzer = visionsdk.ImageAnalyzer(service_options, vision_source, analysis_options)
    result = image_analyzer.analyze()

    if result.reason == visionsdk.ImageAnalysisResultReason.ANALYZED:
        return_res = {'Caption': result.caption.content,
                      'Dense Captions': '; '.join([i.content for i in result.dense_captions]),
                      'Tags': '; '.join([i.name for i in result.tags]),
                      'OCR text': '; '.join([i.content for i in result.text.lines]),
                      'Objects': [{'object_name': object.name, 'bounding_box': (object.bounding_box.x, object.bounding_box.y, object.bounding_box.w, object.bounding_box.h), 'confidence': object.confidence} for object in result.objects], 
                    #   'Dense Captions detail': [{'caption':caption.content, 'bounding_box': (caption.bounding_box.x, caption.bounding_box.y, caption.bounding_box.w, caption.bounding_box.h), 'Confidence': caption.confidence} for caption in result.dense_captions]
                      }
        return return_res
    else:
        error_details = visionsdk.ImageAnalysisErrorDetails.from_result(result)
        print(" Analysis failed.")
        print("   Error reason: {}".format(error_details.reason))
        print("   Error code: {}".format(error_details.error_code))
        print("   Error message: {}".format(error_details.message))
        return

@retry(stop_max_attempt_number=6, wait_random_min=500, wait_random_max=1000, stop_max_delay=60000)
def getImgInfo(img_path, img_root):
    img_realpath = os.path.join(img_root, img_path)

    if not os.path.exists(img_realpath):
        print('img不存在：{}'.format(img_realpath))
        return 'There is no relevant image information.'
    
    return_res = azure_analysisImg(img_realpath)
    return return_res


def dumpdata(obj, path_json):
    with open(path_json, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=4)
    return

def ex_dataset(dataset, save_path, temp_path):
    for idx, (sample_id, sample) in tqdm(enumerate(dataset.items())):
        if 'mention_imgdesc_Azure' in sample and sample['mention_imgdesc_Azure'] not in [{}, None, ]:
            continue
        
        mention_imgpath = sample['mention_imgpath']
        try:
            img_info = getImgInfo(mention_imgpath, img_root='./dataset_wikidiverse/mentionIMG/')
            dataset[sample_id]['mention_imgdesc_Azure'] = img_info
        except Exception as e:
            print(e)
            time.sleep(20)

        if idx % 10 == 0:
            dumpdata(dataset, temp_path)
    dumpdata(dataset, save_path)
    return
