import torch
import torch.nn as nn
import numpy as np

import _init_paths
from core.model import get_pose_net 
from config import cfg
from einops import rearrange


# from thop import profile
from torchstat import stat
from thop import profile
# import torchsummary as summary

from torchsummary import summary
import os
import torch.nn.functional as F
from core.inference import get_max_preds
from core.create_adj import adj_mx_from_skeleton
# from core.evaluate import accuracy
os.environ["CUDA_VISIBLE_DEVICES"] = "0"



if __name__ == '__main__':
    
    
    # x = torch.rand(1, 3, 5, 5).cuda()
    # _, _, h, w = x.shape
    # print(x)
    # x = rearrange(x, 'b c h w -> b (h w) c')
    # print(x)
    # x = rearrange(x, 'b (h w) c -> b c h w', h = h, w = w)
    # print(x)

    x = torch.rand(2, 3, 256, 192).cuda()
    model = get_pose_net(cfg, True).cuda()
    x, y= model(x)
    print(x.shape)
    print(y.shape)

    dummy_input = torch.randn(1, 3, 256,192).cuda()
    model_ = get_pose_net(cfg, True).cuda()
    flops, params = profile(model_, (dummy_input,))
    print('flops: ', flops, 'params: ', params)
    print('flops: %.2f M, params: %.2f M' % (flops / 1000000.0, params / 1000000.0))