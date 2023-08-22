# MAET
This is the official repository of our paper for ACM MM 23: Multimodal Adaptive Emotion Transformer with Flexible Modality Inputs on A Novel Dataset with Continuous Labels

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

```