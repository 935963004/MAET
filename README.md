# MAET
This is the official repository of our paper for ACM MM 2023: [Multimodal Adaptive Emotion Transformer with Flexible Modality Inputs on A Novel Dataset with Continuous Labels](https://dl.acm.org/doi/10.1145/3581783.3613797).

NEWS: The extended version of this conference paper has been accepted by IEEE Transactions on Affective Computing. Please refer to [SEED-VII: A Multimodal Dataset of Six Basic Emotions with Continuous Labels for Emotion Recognition](https://xploreqa.ieee.org/document/10731546). The dataset is now available.

## Abstract
Emotion recognition from physiological signals is a topic of widespread interest, and researchers continue to develop novel techniques for perceiving emotions. However, the emergence of deep learning has highlighted the need for high-quality emotional datasets to accurately decode human emotions. In this study, we present a novel multimodal emotion dataset that incorporates electroencephalography (EEG) and eye movement signals to systematically explore human emotions. Seven basic emotions (happy, sad, fear, disgust, surprise, anger, and neutral) are elicited by a large number of 80 videos and fully investigated with continuous labels that indicate the intensity of the corresponding emotions. Additionally, we propose a novel Multimodal Adaptive Emotion Transformer (MAET), that can flexibly process both unimodal and multimodal inputs. Adversarial training is utilized in MAET to mitigate subject discrepancy, which enhances domain generalization. Our extensive experiments, encompassing both subject-dependent and cross-subject conditions, demonstrate MAET's superior performance in handling various inputs. The filtering of data for high emotional evocation using continuous labels proved to be effective in the experiments. Furthermore, the complementary properties between EEG and eye movements are observed.

## Dataset
The SEED-VII dataset is publicly available on [here](https://bcmi.sjtu.edu.cn/home/seed/seed-vii.html).

## Extract Features
- EEG_feature_extractor/extract_EEG_features.py: Extract EEG features
- Eye_feature_extractor/extract_eye_features.py: Extract eye movement features

## Requirements
* python==3.10.9
* pytorch==2.0.0
* timm==0.4.12

## Example
Example code for the use of MAET:
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
Example code for cross-subject training:
```python
import torch
import torch.nn.functional as F
from model import MAET
from functools import partial
from sklearn.metrics import accuracy_score
import math


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, epsilon: float = 0.1, reduction='mean'):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction

    def forward(self, preds, target):
        n = preds.size()[-1]
        log_preds = F.log_softmax(preds, dim=-1)
        loss = reduce_loss(-log_preds.sum(dim=-1), self.reduction)
        nll = F.nll_loss(log_preds, target, reduction=self.reduction)
        return linear_combination(loss / n, nll, self.epsilon)


def train(optimizer, train_dataloader, local_rank, epochs, num_domains):
    model = MAET(embed_dim=32, num_classes=7, eeg_seq_len=5, eye_seq_len=5, eeg_dim=310, eye_dim=33, depth=3, num_heads=4, qkv_bias=True, mixffn_start_layer_index=2, norm_layer=partial(nn.LayerNorm, eps=1e-6), domain_generalization=True, num_domains=num_domains)
    criterion = LabelSmoothingCrossEntropy()
    for epoch in range(epochs):
        model.train()
        alpha = 2 / (1 + math.exp(-10 * epoch / epochs)) - 1
        label_smoothing = (num_domains - 1) / num_domains * epoch / epochs
        criterion.epsilon = label_smoothing
        loss_all = 0
        preds = []
        labels = []
        for eeg, label, domain_label in train_dataloader:
            label = label.to(local_rank, non_blocking=True)
            eeg = eeg.to(local_rank, non_blocking=True)
            domain_label = domain_label.to(local_rank, non_blocking=True)
            outputs, domain_output = model(eeg=eeg, alpha_=alpha)
            loss_ce = F.cross_entropy(input=outputs, target=label.long())
            loss_domain = criterion(domain_output, domain_label.long())
            loss = loss_ce + loss_domain
            loss_all += loss.item()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            preds.append(torch.argmax(outputs, dim=-1).cpu())
            labels.append(label.cpu())
        pred = torch.cat(preds, dim=0)
        label = torch.cat(labels, dim=0)
        train_accuracy = accuracy_score(label, pred)
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
keywords = {eye movements, emotion recognition, eeg, dataset, continuous label},
location = {Ottawa ON, Canada},
series = {MM '23}
}
```
or
```
@ARTICLE{10731546,
  author={Jiang, Wei-Bang and Liu, Xuan-Hao and Zheng, Wei-Long and Lu, Bao-Liang},
  journal={IEEE Transactions on Affective Computing}, 
  title={SEED-VII: A Multimodal Dataset of Six Basic Emotions With Continuous Labels for Emotion Recognition}, 
  year={2025},
  volume={16},
  number={2},
  pages={969-985},
  keywords={Electroencephalography;Emotion recognition;Brain modeling;Physiology;Videos;Electrocardiography;Transformers;Recording;Computational modeling;Affective computing;Basic emotions;continuous label;EEG;emotion recognition;eye movements;multimodal dataset},
  doi={10.1109/TAFFC.2024.3485057}}
```
