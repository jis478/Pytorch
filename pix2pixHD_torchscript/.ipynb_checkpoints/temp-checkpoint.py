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

if __name__ == '__main__':
    
    opt = TestOptions().parse(save=False)
    print(opt)
    model = create_model(opt)
    model = model.cuda()
    traced_script_module = torch.jit.script(model)
    traced_script_module.save('model_torchscript.pt')
    print('TorchScript Conversion Finished!')
    
