import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.ops import DeformConv2d

if __name__=='__main__':
    filter = torch.ones(1,1,3,3).cuda()
    dcn = DeformConv2d(1,1,kernel_size=3,padding=3,dilation=3).cuda()
    input = torch.rand(1,1,11,10).cuda()
    # print(input)
    offset = -torch.ones((1,18,11,10)).cuda().contiguous()
    offset[:,::2,:,:] -= 1
    dcn.weight.data.copy_(filter)
    output = dcn(input, offset)
    print(output.shape)