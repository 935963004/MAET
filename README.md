# MAET
This is the official repository of our paper for ACM MM 23: [Multimodal Adaptive Emotion Transformer with Flexible Modality Inputs on A Novel Dataset with Continuous Labels](https://dl.acm.org/doi/10.1145/3581783.3613797)

## Abstract
Emotion recognition from physiological signals is a topic of widespread interest, and researchers continue to develop novel techniques for perceiving emotions. However, the emergence of deep learning has highlighted the need for high-quality emotional datasets to accurately decode human emotions. In this study, we present a novel multimodal emotion dataset that incorporates electroencephalography (EEG) and eye movement signals to systematically explore human emotions. Seven basic emotions (happy, sad, fear, disgust, surprise, anger, and neutral) are elicited by a large number of 80 videos and fully investigated with continuous labels that indicate the intensity of the corresponding emotions. Additionally, we propose a novel Multimodal Adaptive Emotion Transformer (MAET), that can flexibly process both unimodal and multimodal inputs. Adversarial training is utilized in MAET to mitigate subject discrepancy, which enhances domain generalization. Our extensive experiments, encompassing both subject-dependent and cross-subject conditions, demonstrate MAET's superior performance in handling various inputs. The filtering of data for high emotional evocation using continuous labels proved to be effective in the experiments. Furthermore, the complementary properties between EEG and eye movements are observed.

## Dataset
The dataset will be publicly available on [here](https://bcmi.sjtu.edu.cn/~seed/index.html) soon.

## Requirements
* python==3.10.9
* pytorch==2.0.0
* timm==0.4.12

## Example
```python
import torch
from torch import nn
from model import MAET
from functools import partial

model = MAET(embed_dim=32, num_classes=7, eeg_seq_len=5, eye_seq_len=5, eeg_dim=310, eye_dim=33, depth=3, num_heads=4, qkv_bias=True, mixffn_start_layer_index=2, norm_layer=partial(nn.LayerNorm, eps=1e-6))

input_eeg = torch.randn(64, 310)
input_eye = torch.randn(64, 33)

# single EEG input
out_eeg = model(eeg=input_eeg)

# single eye movements input
out_eye = model(eye=input_eye)

# multimodal input
out_mul = model(eeg=input_eeg, eye=input_eye)
```

## Citation
If you find our paper/code/dataset useful, please consider citing our work:
```
@inproceedings{10.1145/3581783.3613797,
author = {Jiang, Wei-Bang and Liu, Xuan-Hao and Zheng, Wei-Long and Lu, Bao-Liang},
title = {Multimodal Adaptive Emotion Transformer with Flexible Modality Inputs on A Novel Dataset with Continuous Labels},
year = {2023},
isbn = {9798400701085},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3581783.3613797},
doi = {10.1145/3581783.3613797},
abstract = {Emotion recognition from physiological signals is a topic of widespread interest, and researchers continue to develop novel techniques for perceiving emotions. However, the emergence of deep learning has highlighted the need for high-quality emotional datasets to accurately decode human emotions. In this study, we present a novel multimodal emotion dataset that incorporates electroencephalography (EEG) and eye movement signals to systematically explore human emotions. Seven basic emotions (happy, sad, fear, disgust, surprise, anger, and neutral) are elicited by a large number of 80 videos and fully investigated with continuous labels that indicate the intensity of the corresponding emotions. Additionally, we propose a novel Multimodal Adaptive Emotion Transformer (MAET), that can flexibly process both unimodal and multimodal inputs. Adversarial training is utilized in MAET to mitigate subject discrepancy, which enhances domain generalization. Our extensive experiments, encompassing both subject-dependent and cross-subject conditions, demonstrate MAET's superior performance in handling various inputs. The filtering of data for high emotional evocation using continuous labels proved to be effective in the experiments. Furthermore, the complementary properties between EEG and eye movements are observed. Our code is available at https://github.com/935963004/MAET.},
booktitle = {Proceedings of the 31st ACM International Conference on Multimedia},
pages = {5975–5984},
numpages = {10},
keywords = {eye movements, dataset, continuous label, eeg, emotion recognition},
location = {Ottawa ON, Canada},
series = {MM '23}
}
```
