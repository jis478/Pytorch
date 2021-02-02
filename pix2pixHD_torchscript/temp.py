import os
from collections import OrderedDict
import easydict
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import util.util as util
from util.visualizer import Visualizer
from util import html
import torch
import numpy as np

# The original torch model path can be found in /Options/Base_Option.py 
TORCHSCROPT_MODEL_PATH = './model_torchscript.pt'

if __name__ == '__main__':
    
    # CAUTION: Only "no instance" case is supported 

    opt = TestOptions().parse(save=False)
    opt.no_instance = True
    model = create_model(opt)
    model = model.cuda()
    traced_script_module = torch.jit.script(model)
    traced_script_module.save(TORCHSCROPT_MODEL_PATH)

    print('TorchScript Conversion Finished!')
    
