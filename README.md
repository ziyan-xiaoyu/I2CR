<br />
<p align="center">
  <h1 align="center"> 🔭 I2CR: Intra- and Inter-modal Collaborative Reflections for Multimodal Entity Linking</h1>
<!--   <h3 align="center">SpatialMQA: A new benchmark dataset for spatial reasoning of MLLMs.</h3> -->
  
  <p align="center">  
<!--     <a href="https://arxiv.org/abs/2205.00363">arxiv</a> -->
    ·
    <a href="https://github.com/ziyan-xiaoyu/SpatialMQA/blob/master/figures/framework.png">framework</a>
    ·
    <a href="https://github.com/ziyan-xiaoyu/I2CR/blob/master/codes/inference/main/run_main.py">codes</a>
<!--     <a href="https://paperswithcode.com/sota/visual-reasoning-on-vsr">benchmark</a> -->
    
  </p>
</p>



## Contents
- [I2CR](#I2CR)
  - [Contents](#contents)
  - [Overview](#overview)
  - [Datasets](#datasets)
  - [Usage](#usage)
  - [Evaluate](#evaluate)

## Overview
<img src="/figures/framework.png"/>

Our proposed I2CR consists of four key steps. (1) Target Entity Selection (TES). (2) Intra-modal Consistency Reflection (ICR). (3) Inter-modal Alignment Verification (IAV). and (4) Visual Iterative Feedback (VIF).


## Datasets
In this study, we employ the test sets from the following three datasets for evaluation: WkiMEL, WikiDiverse and Richpedia-MEL.
The processed input datasets are stored in `./dataset_WikiMEL`, `./dataset_WikiDiverse` and `./dataset_Richpedia`. 
These files also contain four distinct visual clues extracted from the images using four different image-to-text models.

Alternatively, you can download the raw WikiMEL and Richpedia-MEL from [https://github.com/seukgcode/MELBench](https://github.com/seukgcode/MELBench), and WikiDiverse from [https://github.com/wangxw5/wikiDiverse](https://github.com/wangxw5/wikidiverse) for . Then use the `./main/get_mention_img_info.py` to obtain different types of visual clues for mention images.


## Usage
### Step 1: Install and set up environment
```
>>> pip install -r requirements.txt
>>> conda create -n I2CR_env python==3.8.20
>>> conda activate I2CR_env
```

**Step 2**: Prepare datasets
<br>
You can prepare the dataset using the methods provided in the previous section `Datasets`.

**Step 3**: Prepare models
<br>
Download the LLM
```
>>> git lfs install
>>> git clone https://huggingface.co/unsloth/llama-3-8b-Instruct models/llama-3-8b-Instruct
```
Download the embedding model
```
>>> git clone https://huggingface.co/Salesforce/SFR-Embedding-Mistral
```
The precomputed scores of the CLIP model are stored in `./codes/inference/visual_expert/output`. 
Alternatively, if you'd like to download CLIP and recalculate the scores on your own, you can use the following command to download the model:
```
>>> git clone https://huggingface.co/laion/CLIP-ViT-bigG-14-laion2B-39B-b160k
```
Then, you can obtain these scores through the `./codes/inference/visual_expert/ve_score.py` script.

**Step 4**: Fine-tune LLM
<br>
We provide a trained checkpoint(`./codes/fine-tune/llama3/model_arg/llama3_8b_checkpoints`), you just need to record its path.
Alternatively, if you want to train a new checkpoint, please refer to llama-factory(https://github.com/hiyouga/LLaMA-Factory) or use peft (https://github.com/huggingface/peft) or swift (https://github.com/modelscope/swift).

**Step 5**: Run
```
>>> cd codes/inference/main/
>>> python run_main.py
```
Please ensure that all paths in params.py, run_main.py, and infer_SFR.py are correct.

## Evaluate
The results are saved in `/dataset_WikiMEL/result`, `/dataset_WikiDiverse/result` and `/dataset_RichMEL/result`.
You can use [`calculate_top1_acc.py/`](codes/inference/tool/calculate_top1_acc.py) to calculate the model's accuracy on different datasets.
