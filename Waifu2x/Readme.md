# Waifu2x multi-GPU (or single-GPU) inference 

## Overview 
- A multi-GPU (or single-GPU) inference added to [Waifu2x re-implementation](https://github.com/yu45020/Waifu2x/blob/master/Models.py) 
- Curretnly only CARN-V2 is supported for multi-GPU inference.
- A single GPU inference is also supported if the number of GPUs sets to 1.
- All the code changes are contained in Custom.py and the original codes haven't been modified. 

## Inference example

```python
import os
import numpy as np
import glob
import time
import shutil
import tqdm
import torch
from torch.utils.data import DataLoader
from torchvision import utils

from utils.prepare_images import *
from Models import *
from Custom import *   # all of the modifications are contained here


# batch size
batch_size=512 

# Image size after super resolution
size = 256 

# GPU No. (same as the input for "CUDA_VISIONABLE_DEVICE")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    
gpu_no = '0,1'  # if you want to use 2 GPUs

# Pre-trained CARN V2 model location
model_path = './model_check_points/CRAN_V2/CARN_model_checkpoint.pt' 

# source (before Super resolution) and destination (after super resolution) image locations
dst = './dst/' 
src = './src/' 

# set-up
gpu_tot = gpusetting(gpu_no) 
img_list = imglist(src, dst)
model = ModelBuild(model_path, device)
dataset = CustomDataset(img_list=img_list, gpu_tot=gpu_tot, dst=dst, size=size)
dataloader = DataLoader(dataset, 
                        batch_size=batch_size,
                        shuffle=False,
                        collate_fn=collate_patches)

# CARN-based super resolution
for batch in tqdm.tqdm(dataloader): 
    with torch.no_grad(): 
        model(batch) 
        
```
