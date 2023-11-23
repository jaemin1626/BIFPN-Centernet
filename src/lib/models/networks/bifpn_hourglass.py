# ------------------------------------------------------------------------------
# This code is base on 
# CornerNet (https://github.com/princeton-vl/CornerNet)
# Copyright (c) 2018, University of Michigan
# Licensed under the BSD 3-Clause License
# ------------------------------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn

class convolution(nn.Module):
    def __init__(self, k, inp_dim, out_dim, stride=1, with_bn=True):
        super(convolution, self).__init__()

        pad = (k - 1) // 2
        self.conv = nn.Conv2d(inp_dim, out_dim, (k, k), padding=(pad, pad), stride=(stride, stride), bias=not with_bn)
        self.bn   = nn.BatchNorm2d(out_dim) if with_bn else nn.Sequential()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        conv = self.conv(x)
        bn   = self.bn(conv)
        relu = self.relu(bn)
        return relu

class fully_connected(nn.Module):
    def __init__(self, inp_dim, out_dim, with_bn=True):
        super(fully_connected, self).__init__()
        self.with_bn = with_bn

        self.linear = nn.Linear(inp_dim, out_dim)
        if self.with_bn:
            self.bn = nn.BatchNorm1d(out_dim)
        self.relu   = nn.ReLU(inplace=True)

    def forward(self, x):
        linear = self.linear(x)
        bn     = self.bn(linear) if self.with_bn else linear
        relu   = self.relu(bn)
        return relu

class residual(nn.Module):
    def __init__(self, k, inp_dim, out_dim, stride=1, with_bn=True):
        super(residual, self).__init__()

        self.conv1 = nn.Conv2d(inp_dim, out_dim, (3, 3), padding=(1, 1), stride=(stride, stride), bias=False)
        self.bn1   = nn.BatchNorm2d(out_dim)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_dim, out_dim, (3, 3), padding=(1, 1), bias=False)
        self.bn2   = nn.BatchNorm2d(out_dim)
        
        self.skip  = nn.Sequential(
            nn.Conv2d(inp_dim, out_dim, (1, 1), stride=(stride, stride), bias=False),
            nn.BatchNorm2d(out_dim)
        ) if stride != 1 or inp_dim != out_dim else nn.Sequential()
        self.relu  = nn.ReLU(inplace=True)

    def forward(self, x):
        conv1 = self.conv1(x)
        bn1   = self.bn1(conv1)
        relu1 = self.relu1(bn1)

        conv2 = self.conv2(relu1)
        bn2   = self.bn2(conv2)

        skip  = self.skip(x)
        return self.relu(bn2 + skip)
class DepthwiseConvBlock(nn.Module):
    """
    Depthwise seperable convolution. 
    
    """
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, freeze_bn=False):
        super(DepthwiseConvBlock,self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, 
                               padding, dilation, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                                   stride=1, padding=0, dilation=1, groups=1, bias=False)
        
        self.bn = nn.BatchNorm2d(out_channels, momentum=0.9997, eps=4e-5)
        self.act = nn.ReLU()
        
    def forward(self, inputs):
        x = self.depthwise(inputs)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.act(x)

class bifpn(nn.Module):
    def __init__(self):
        super(bifpn, self).__init__()
        self.p6_td = DepthwiseConvBlock(384,384)
        self.p5_td = DepthwiseConvBlock(384,384)
        self.p4_td = DepthwiseConvBlock(384,384)
        self.p3_td = DepthwiseConvBlock(256,256)
        self.p2_td = DepthwiseConvBlock(256,256)

        self.p3_out = DepthwiseConvBlock(256,256)
        self.p4_out = DepthwiseConvBlock(384,384)
        self.p5_out = DepthwiseConvBlock(384,384)
        self.p6_out = DepthwiseConvBlock(384,384)
        self.p7_out = DepthwiseConvBlock(512,512)

        self.p7 = nn.Conv2d(512,384,1,1,0)
        self.p6 = nn.Conv2d(384,384,1,1,0)
        self.p5 = nn.Conv2d(384,384,1,1,0)
        self.p4 = nn.Conv2d(384,256,1,1,0)
        self.p3 = nn.Conv2d(256,256,1,1,0)
        
        self.p7_ = nn.Conv2d(384,512,1,2,0)
        self.p6_ = nn.Conv2d(384,384,1,2,0)
        self.p5_ = nn.Conv2d(384,384,1,2,0)
        self.p4_ = nn.Conv2d(256,384,1,2,0)
        self.p3_ = nn.Conv2d(256,256,1,2,0)

        self.w1 =  nn.Parameter(torch.ones((2,5), dtype=torch.float32), requires_grad=True)
        self.w2 =  nn.Parameter(torch.ones((3,5), dtype=torch.float32), requires_grad=True)
        
        self.w1_relu = nn.ReLU()
        self.w2_relu = nn.ReLU()

        self.epsilon = 0.0001
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
    
    def forward(self,low,low_list):
        
        w1 = self.w1_relu(self.w1)
        w1 /= torch.sum(w1, dim=0) + self.epsilon
        
        w2 = self.w2_relu(self.w2)
        w2 /= torch.sum(w2, dim=0) + self.epsilon

        
        ####### down ##############
        p7_x = low_list[5]
        p7   = self.upsample(self.p7(low_list[5]))
        p6_  = self.p6_td(low_list[4] * w1[0,0]  + w1[1,0] * p7)
        
        p6   = self.upsample(self.p6(p6_))
        p5_  = self.p5_td(low_list[3] * w1[0,1]  + w1[1,1] * p6)
        
        p5   = self.upsample(self.p5(p5_))
        p4_  = self.p4_td(low_list[2] * w1[0,2]  + w1[1,2] * p5)
        
        p4   = self.upsample(self.p4(p4_))
        p3_  = self.p3_td(low_list[1] * w1[0,3]  + w1[1,3] * p4)

        p3   = self.upsample(self.p3(p3_))
        p2_  = self.p2_td(low_list[0] * w1[0,4] + w1[1,4]  * p3)
        
        #### Top ####################
        p3_out  = self.p3_(p2_)
        p3_out_ = self.p3_out(w2[0,0] * low_list[1] + w2[1,0] * p3_ + w2[2,0] * p3_out)

        p4_out  = self.p4_(p3_out_)
        p4_out_ = self.p4_out(w2[0,1] * low_list[2] + w2[1,1] * p4_ + w2[2,1] * p4_out)

        p5_out  = self.p5_(p4_out_)
        p5_out_ = self.p5_out(w2[0,2] * low_list[3] + w2[1,2] * p5_ + w2[2,2] * p5_out)

        p6_out  = self.p6_(p5_out_)
        p6_out_ = self.p6_out(w2[0,3] * low_list[4] + w2[1,3] * p6_ + w2[2,3] * p6_out)

        p7_out  = self.p7_(p6_out_)
        p7_out_ = self.p7_out(w2[0,4] * low_list[5] + w2[1,4] * p7_x + w2[2,4] * p7_out)

        return p7_out_, [ p2_, p3_out_, p4_out_, p5_out_, p6_out_, p7_out_ ]

def make_layer(k, inp_dim, out_dim, modules, layer=convolution, **kwargs):
    layers = [layer(k, inp_dim, out_dim, **kwargs)]
    for _ in range(1, modules):
        layers.append(layer(k, out_dim, out_dim, **kwargs))
    return nn.Sequential(*layers)

def make_layer_revr(k, inp_dim, out_dim, modules, layer=convolution, **kwargs):
    layers = []
    for _ in range(modules - 1):
        layers.append(layer(k, inp_dim, inp_dim, **kwargs))
    layers.append(layer(k, inp_dim, out_dim, **kwargs))
    return nn.Sequential(*layers)

class MergeUp(nn.Module):
    def forward(self, up1, up2):
        return up1 + up2

def make_merge_layer(dim):
    return MergeUp()

# def make_pool_layer(dim):
#     return nn.MaxPool2d(kernel_size=2, stride=2)

def make_pool_layer(dim):
    return nn.Sequential()

def make_unpool_layer(dim):
    return nn.Upsample(scale_factor=2)

def make_kp_layer(cnv_dim, curr_dim, out_dim):
    return nn.Sequential(
        convolution(3, cnv_dim, curr_dim, with_bn=False),
        nn.Conv2d(curr_dim, out_dim, (1, 1))
    )

def make_inter_layer(dim):
    return residual(3, dim, dim)

def make_cnv_layer(inp_dim, out_dim):
    return convolution(3, inp_dim, out_dim)

class kp_module(nn.Module):
    def __init__(
        self,current_stack, n, dims, modules, layer=residual,
        make_up_layer=make_layer, make_low_layer=make_layer,
        make_hg_layer=make_layer, make_hg_layer_revr=make_layer_revr,
        make_pool_layer=make_pool_layer, make_unpool_layer=make_unpool_layer,
        bifpn_layer = bifpn, make_merge_layer=make_merge_layer, **kwargs
    ):
        super(kp_module, self).__init__()

        self.n_  = n
        self.n   = n
        self.current_stack = current_stack
        curr_mod = modules[0]
        next_mod = modules[1]
        
        self.low_list = []

        curr_dim = dims[0]
        next_dim = dims[1]

        self.up1  = make_up_layer(
            3, curr_dim, curr_dim, curr_mod, 
            layer=layer, **kwargs
        )  
        self.max1 = make_pool_layer(curr_dim)
        self.low1 = make_hg_layer(
            3, curr_dim, next_dim, curr_mod,
            layer=layer, **kwargs
        )
        if n>1:
            self.low2 = kp_module(
                self.current_stack, n - 1, dims[1:], modules[1:], layer=layer, 
                make_up_layer=make_up_layer, 
                make_low_layer=make_low_layer,
                make_hg_layer=make_hg_layer,
                make_hg_layer_revr=make_hg_layer_revr,
                make_pool_layer=make_pool_layer,
                make_unpool_layer=make_unpool_layer,
                make_merge_layer=make_merge_layer,
                **kwargs
            ) 
        elif self.current_stack==0 and n==1:
            self.low2 = bifpn_layer()
                
        else:
            self.low2 = make_low_layer(3, next_dim, next_dim, next_mod,layer=layer, **kwargs)

        self.low3 = make_hg_layer_revr(
            3, next_dim, curr_dim, curr_mod,
            layer=layer, **kwargs
        )

        self.up2  = make_unpool_layer(curr_dim)
        self.merge = make_merge_layer(curr_dim)

    def forward(self, x, low_list):
        
        if(self.n == 5):
            low_list.append(x)

        up1  = self.up1(x) ## conv
        max1 = self.max1(x)
        low1 = self.low1(max1) ## 256,64,64
        low_list.append(low1)
        
        if(self.n==1 and self.current_stack==1):
            low2 = self.low2(low1)
        else:
            low2, low_list = self.low2(low1, low_list) 
        
        low3 = self.low3(low2)
        up2  = self.up2(low3)

        return self.merge(low_list[5-self.n], up2), low_list

class exkp(nn.Module):
    def __init__(
        self, n, nstack, dims, modules, heads, pre=None, cnv_dim=256, 
        make_tl_layer=None, make_br_layer=None,
        make_cnv_layer=make_cnv_layer, make_heat_layer=make_kp_layer,
        make_tag_layer=make_kp_layer, make_regr_layer=make_kp_layer,
        make_up_layer=make_layer, make_low_layer=make_layer, 
        make_hg_layer=make_layer, make_hg_layer_revr=make_layer_revr,
        make_pool_layer=make_pool_layer, make_unpool_layer=make_unpool_layer,
        make_merge_layer=make_merge_layer, make_inter_layer=make_inter_layer, 
        kp_layer=residual
    ):
        super(exkp, self).__init__()

        self.nstack    = nstack
        self.heads     = heads

        curr_dim = dims[0]

        self.pre = nn.Sequential(
            convolution(7, 3, 128, stride=2),
            residual(3, 128, 256, stride=2)
        ) if pre is None else pre

        self.kps  = nn.ModuleList([
            kp_module(
                current_stack,n, dims, modules, layer=kp_layer,
                make_up_layer=make_up_layer,
                make_low_layer=make_low_layer,
                make_hg_layer=make_hg_layer,
                make_hg_layer_revr=make_hg_layer_revr,
                make_pool_layer=make_pool_layer,
                make_unpool_layer=make_unpool_layer,
                make_merge_layer=make_merge_layer
            ) for current_stack in range(nstack)
        ])
        self.cnvs = nn.ModuleList([
            make_cnv_layer(curr_dim, cnv_dim) for _ in range(nstack)
        ])

        self.inters = nn.ModuleList([
            make_inter_layer(curr_dim) for _ in range(nstack - 1)
        ])

        self.inters_ = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(curr_dim, curr_dim, (1, 1), bias=False),
                nn.BatchNorm2d(curr_dim)
            ) for _ in range(nstack - 1)
        ])
        self.cnvs_   = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(cnv_dim, curr_dim, (1, 1), bias=False),
                nn.BatchNorm2d(curr_dim)
            ) for _ in range(nstack - 1)
        ])
        ## keypoint heatmaps
        for head in heads.keys():
            if 'hm' in head:
                module =  nn.ModuleList([
                    make_heat_layer(
                        cnv_dim, curr_dim, heads[head]) for _ in range(nstack)
                ])
                self.__setattr__(head, module)
                for heat in self.__getattr__(head):
                    heat[-1].bias.data.fill_(-2.19)
            else:
                module = nn.ModuleList([
                    make_regr_layer(
                        cnv_dim, curr_dim, heads[head]) for _ in range(nstack)
                ])
                self.__setattr__(head, module)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, image):
        # print('image shape', image.shape)
        inter = self.pre(image)
        outs  = []
        for ind in range(self.nstack):
            kp_, cnv_  = self.kps[ind], self.cnvs[ind]
            kp  = kp_(inter,[])
            cnv = cnv_(kp[0])

            out = {}
            for head in self.heads:
                layer = self.__getattr__(head)[ind]
                y = layer(cnv)
                out[head] = y
            
            outs.append(out)
            if ind < self.nstack - 1:
                inter = self.inters_[ind](inter) + self.cnvs_[ind](cnv)
                inter = self.relu(inter)
                inter = self.inters[ind](inter)
        return outs

def make_hg_layer(kernel, dim0, dim1, mod, layer=convolution, **kwargs):
    layers  = [layer(kernel, dim0, dim1, stride=2)]
    layers += [layer(kernel, dim1, dim1) for _ in range(mod - 1)]
    return nn.Sequential(*layers)

class HourglassNet(exkp):
    def __init__(self, heads, num_stacks=1):
        n       = 5
        dims    = [256, 256, 384, 384, 384, 512]
        modules = [2, 2, 2, 2, 2, 4]

        super(HourglassNet, self).__init__(
            n, num_stacks, dims, modules, heads,
            make_tl_layer=None,
            make_br_layer=None,
            make_pool_layer=make_pool_layer,
            make_hg_layer=make_hg_layer,
            kp_layer=residual, cnv_dim=256
        )

def get_bifpn_hourglass(num_layers, heads, head_conv):
  model = HourglassNet(heads, 2)
  return model