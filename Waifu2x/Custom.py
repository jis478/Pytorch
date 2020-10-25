
# torch
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.modules import Module
from torch.nn.parallel.scatter_gather import gather
from torch.nn.parallel.replicate import replicate
from torch.nn.parallel.parallel_apply import parallel_apply
from torch.cuda._utils import _get_device_index
from torch.utils.data import Dataset

# general
import operator
import warnings
from itertools import chain
import glob
import numpy as np
import os
from PIL import Image
import shutil

# codes
from Models import CARN, CARN_Block, ConvBlock, BN_convert_float
from utils.prepare_images import ImageSplitter 
        
def _check_balance(device_ids):
    imbalance_warn = """
    There is an imbalance between your GPUs. You may want to exclude GPU {} which
    has less than 75% of the memory or cores of GPU {}. You can do so by setting
    the device_ids argument to DataParallel, or by setting the CUDA_VISIBLE_DEVICES
    environment variable."""
    device_ids = list(map(lambda x: _get_device_index(x, True), device_ids))
    dev_props = [torch.cuda.get_device_properties(i) for i in device_ids]

    def warn_imbalance(get_prop):
        values = [get_prop(props) for props in dev_props]
        min_pos, min_val = min(enumerate(values), key=operator.itemgetter(1))
        max_pos, max_val = max(enumerate(values), key=operator.itemgetter(1))
        if min_val / max_val < 0.75:
            warnings.warn(imbalance_warn.format(device_ids[min_pos], device_ids[max_pos]))
            return True
        return False

    if warn_imbalance(lambda props: props.total_memory):
        return
    if warn_imbalance(lambda props: props.multi_processor_count):
        return


class DataParallel(Module):
    def __init__(self, module, device_ids=None, output_device=None, dim=0):
        super(DataParallel, self).__init__()

        if not torch.cuda.is_available():
            self.module = module
            self.device_ids = []
            return

        if device_ids is None:
            device_ids = list(range(torch.cuda.device_count()))
        if output_device is None:
            output_device = device_ids[0]

        self.dim = dim
        self.module = module
        self.device_ids = list(map(lambda x: _get_device_index(x, True), device_ids))
        self.output_device = _get_device_index(output_device, True)
        self.src_device_obj = torch.device("cuda:{}".format(self.device_ids[0]))

        _check_balance(self.device_ids)

        if len(self.device_ids) == 1:
            self.module.cuda(device_ids[0])

    def forward(self, *inputs, **kwargs):
        if not self.device_ids:
            return self.module(*inputs, **kwargs)

        for t in chain(self.module.parameters(), self.module.buffers()):
            if t.device != self.src_device_obj:
                raise RuntimeError("module must have its parameters and buffers "
                                   "on device {} (device_ids[0]) but found one of "
                                   "them on device: {}".format(self.src_device_obj, t.device))
        inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids)
        
        if len(self.device_ids) == 1:
            return self.module(*inputs[0], **kwargs[0])
        replicas = self.replicate(self.module, self.device_ids[:len(inputs)])
        outputs = self.parallel_apply(replicas, inputs, kwargs)

        return self.gather(outputs, self.output_device)

    def replicate(self, module, device_ids):
        return replicate(module, device_ids, not torch.is_grad_enabled())

    def scatter(self, inputs, kwargs, device_ids):
        return scatter_kwargs(inputs, kwargs, device_ids, dim=self.dim)

    def parallel_apply(self, replicas, inputs, kwargs):
        return parallel_apply(replicas, inputs, kwargs, self.device_ids[:len(replicas)])

    def gather(self, outputs, output_device):
        return gather(outputs, output_device, dim=self.dim)

      
        
        
def scatter(inputs, target_gpus, dim=0):
    r"""
    Slices tensors into approximately equal chunks and
    distributes them across given GPUs. Duplicates
    references to objects that are not tensors.
    """
    def scatter_map(obj):
        if isinstance(obj, torch.Tensor):
            return Scatter.apply(target_gpus, None, dim, obj)
        if isinstance(obj, tuple) and len(obj) > 0:
            return list(zip(*map(scatter_map, obj)))
        if isinstance(obj, list) and len(obj) > 0:
            size = len(obj) // len(target_gpus)
            return [obj[i * size:(i + 1) * size] for i in range(len(target_gpus))]
        if isinstance(obj, dict) and len(obj) > 0:
            return list(map(type(obj), zip(*map(scatter_map, obj.items()))))
        return [obj for targets in target_gpus]
    try:
        return scatter_map(inputs)
    finally:
        scatter_map = None


def scatter_kwargs(inputs, kwargs, target_gpus, dim=0):
    r"""Scatter with support for kwargs dictionary"""
    inputs = scatter(inputs, target_gpus, dim) if inputs else []
    kwargs = scatter(kwargs, target_gpus, dim) if kwargs else []
    if len(inputs) < len(kwargs):
        inputs.extend([() for _ in range(len(kwargs) - len(inputs))])
    elif len(kwargs) < len(inputs):
        kwargs.extend([{} for _ in range(len(inputs) - len(kwargs))])
    inputs = tuple(inputs)
    kwargs = tuple(kwargs)
    return inputs, kwargs



class CustomDataset(Dataset):
    """ Super resolution patch dataset """
    def __init__(self, img_list, device_ids, dst, size):
        """
        Args:
            img_list (string): Path to images on which CARN_v2 model will be applied
            gpu_tot (int): # of GPUs for inference. Here it is used to decide torch data type. 
            dst: destional folder for super resolution images
            size: final image size after super resolution 
        """
        self.img_list = img_list  
        self.device_ids = device_ids
        self.dst = dst
        self.size = size

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()            
        img = Image.open(self.img_list[idx]).convert("RGB")
        img = np.array(img)            
        img_splitter = ImageSplitter(seg_size=64, scale_factor=2, boarder_pad_size=3)
        filename = '_'.join([self.img_list[idx].split('/')[-3], self.img_list[idx].split('/')[-2], self.img_list[idx].split('/')[-1]])
        img_splitter.filename = os.path.join(self.dst, filename) 
        img_splitter.size = self.size
        img_patches = img_splitter.split_img_tensor(img, scale_method=None, img_pad=0)       
        return img_patches, img_splitter
           
        
        
def gpusetting(gpu_no):
    ''' GPU setting ''' 

    device_ids = []
    for i in gpu_no:
        try:
            torch.cuda.get_device_name(int(i))
            device_ids.append(i)
        except:
            continue
    assert len(device_ids) > 0, "No GPU found error"
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"    
    os.environ["CUDA_VISIBLE_DEVICES"] = '{}'.format(','.join(device_ids))
    print('Total {} GPU(s) (No. [{}]) will be used'.format(len(device_ids), ','.join(device_ids)))
    device_ids = [int(id) for id in device_ids]
    return device_ids

            

def imglist(src, dst):
    if os.path.exists(dst):
        shutil.rmtree(dst)
    os.makedirs(dst)
    img_list = glob.glob(src + '*/*/*.jpg') 
    return img_list
    
    
    
def collate_patches(batch):
    """ collate_fn for variable input size """
    return [b for b in batch]



def ModelBuild(model_path, device_ids):    
    """ model building (distributed) """
    model_cran_v2 = CARN_V2_multiGPU(color_channels=3, mid_channels=64, conv=nn.Conv2d,
                        single_conv_size=3, single_conv_group=1,
                        scale=2, activation=nn.LeakyReLU(0.1),
                        SEBlock=True, repeat_blocks=3, atrous=(1, 1, 1))
    model_cran_v2 = network_to_half(model_cran_v2)
    model_cran_v2.load_state_dict(torch.load(model_path))
    model_cran_v2 = DataParallel(model_cran_v2, device_ids) 
    model_cran_v2.to('cuda')
    return model_cran_v2


def SR(model, dataloader, size, dst):
    start_idx = 0
    s = time.time()
    for batch in dataloader: 
        with torch.no_grad(): 
            model(batch) 
            
            
class tofp16_gpu(nn.Module):
    def __init__(self):
        super(tofp16_gpu, self).__init__()

    def forward(self, input):
        return input

    
    
def network_to_half(network):
    return nn.Sequential(tofp16_gpu(), BN_convert_float(network.half()))



class CARN_V2_multiGPU(CARN):
    def __init__(self, color_channels=3, mid_channels=64,
                 scale=2, activation=nn.LeakyReLU(0.1),
                 SEBlock=True, conv=nn.Conv2d,
                 atrous=(1, 1, 1), repeat_blocks=3,
                 single_conv_size=3, single_conv_group=1):
        super(CARN_V2_multiGPU, self).__init__(color_channels=color_channels, mid_channels=mid_channels, scale=scale,
                                      activation=activation, conv=conv)
        num_blocks = len(atrous)
        m = []
        for i in range(num_blocks):
            m.append(CARN_Block(mid_channels, kernel_size=3, padding=1, dilation=1,
                                activation=activation, SEBlock=SEBlock, conv=conv, repeat=repeat_blocks,
                                single_conv_size=single_conv_size, single_conv_group=single_conv_group))
        self.blocks = nn.Sequential(*m)
        self.singles = nn.Sequential(
            *[ConvBlock(mid_channels * (i + 2), mid_channels, kernel_size=single_conv_size,
                        padding=(single_conv_size - 1) // 2, groups=single_conv_group,
                        activation=activation, conv=conv)
              for i in range(num_blocks)])
        
    def sr(self, x):
        x = self.entry_block(x)
        c0 = x
        res = x
        for block, single in zip(self.blocks, self.singles):
            b = block(x)
            c0 = c = torch.cat([c0, b], dim=1)
            x = single(c)
        x = x + res
        x = self.upsampler(x)
        out = self.exit_conv(x)
        return out

    def forward(self, x):  
        device = 'cuda:{}'.format(str(torch.cuda.current_device()))
        for patches, splitter in x: # each loop processes an image
            out = splitter.merge_img_tensor([self.sr(patch.to(device).half()) for patch in patches])
            out = torch.squeeze(F.interpolate(out, size=(splitter.size,splitter.size)).mul_(255).add_(0.5).clamp_(0, 255),0)
            out = Image.fromarray(out.permute(1,2,0).to('cpu', torch.uint8).numpy())
            out.save(splitter.filename)

    
#     def forward(self, x):  
#         device = 'cuda:{}'.format(str(torch.cuda.current_device()))
#         idx = 0
#         return_out = None
#         for patches, splitter in x: # each loop processes an image
#             if return_out is None:
#                 return_out = torch.empty(len(x), 3, splitter.size, splitter.size)
#             out = splitter.merge_img_tensor([self.sr(patch.to(device).half()) for patch in patches])
#             out = torch.squeeze(F.interpolate(out, size=(splitter.size,splitter.size)).mul_(255).add_(0.5).clamp_(0, 255),0)
#             return_out[idx, :] = out
#             idx += 1 
#         return return_out.to(device)
            
