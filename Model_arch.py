# --- Imports --- #
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pdb import set_trace as stx
import math

import onnx
from onnx import shape_inference
import netron


##########################################################################
##---------- FIB Feature interaction block ----------
###    Feature interaction block  #####

#AFF
#ChannelInteractionBlock  represents the primary function of the block, which is to capture and represent the interactions among different channels of the input features.
class FIB(nn.Module):
    #
    def __init__(self, in_channels, height=3,reduction=8,bias=False):
        super(FIB, self).__init__()
        
        self.height = height  # 这里应该是2
        d = max(int(in_channels/reduction),4)  # 维度最大去4
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 自适应平均池化
        self.conv_du = nn.Sequential(nn.Conv2d(in_channels, d, 1, padding=0, bias=bias), nn.LeakyReLU(0.2)) # 输出d维

        self.fcs = nn.ModuleList([])
        for i in range(self.height):
            self.fcs.append(nn.Conv2d(d, in_channels, kernel_size=1, stride=1,bias=bias))
        
        self.softmax = nn.Softmax(dim=1)  # softmax操作

    def forward(self, inp_feats):
        batch_size = inp_feats[0].shape[0]  # batch_size
        n_feats =  inp_feats[0].shape[1] # 特征通道数
        

        inp_feats = torch.cat(inp_feats, dim=1)
        inp_feats = inp_feats.view(batch_size, self.height, n_feats, inp_feats.shape[2], inp_feats.shape[3])
        
        feats_U = torch.sum(inp_feats, dim=1)  # 降维（横向压缩）
        feats_S = self.avg_pool(feats_U) # 平均池化（得到向量）
        feats_Z = self.conv_du(feats_S) # 1×1卷积映射

        attention_vectors = [fc(feats_Z) for fc in self.fcs]  # 得到注意力向量
        attention_vectors = torch.cat(attention_vectors, dim=1) # 降维（横向压缩）
        attention_vectors = attention_vectors.view(batch_size, self.height, n_feats, 1, 1)
        # stx()
        attention_vectors = self.softmax(attention_vectors) # 归一化到0-1区间
        
        feats_V = torch.sum(inp_feats*attention_vectors, dim=1) # 注意力之后的结果
        
        return feats_V        


# GCNet：当Non-local遇见SENet：https://zhuanlan.zhihu.com/p/64988633
class ContextBlock(nn.Module):

    def __init__(self, n_feat, bias=False):
        super(ContextBlock, self).__init__()

        self.conv_mask = nn.Conv2d(n_feat, 1, kernel_size=1, bias=bias)  # 输入是中间层特征通道数 输出通道是1
        self.softmax = nn.Softmax(dim=2)

        self.channel_add_conv = nn.Sequential(
            nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=bias),
            nn.LeakyReLU(0.2),
            nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=bias)
        )

    def modeling(self, x):
        batch, channel, height, width = x.size()
        input_x = x
        # [N, C, H * W]
        input_x = input_x.view(batch, channel, height * width)
        # [N, 1, C, H * W]
        input_x = input_x.unsqueeze(1)
        # [N, 1, H, W]
        context_mask = self.conv_mask(x)
        # [N, 1, H * W]
        context_mask = context_mask.view(batch, 1, height * width)
        # [N, 1, H * W]
        context_mask = self.softmax(context_mask)
        # [N, 1, H * W, 1]
        context_mask = context_mask.unsqueeze(3)
        # [N, 1, C, 1]
        context = torch.matmul(input_x, context_mask)
        # [N, C, 1, 1]
        context = context.view(batch, channel, 1, 1)

        return context

    def forward(self, x):
        # [N, C, 1, 1]
        context = self.modeling(x)

        # [N, C, 1, 1]
        channel_add_term = self.channel_add_conv(context)
        x = x + channel_add_term

        return x

##########################################################################
### --------- Feature enhancement block ----------



class FEB(nn.Module):
    def __init__(self, n_feat, kernel_size=3, reduction=8, bias=False, groups=1):
        super(FEB, self).__init__()
        
        act = nn.LeakyReLU(0.2)

        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat, kernel_size=3, stride=1, padding=1, bias=bias, groups=groups),
            act, 
            nn.Conv2d(n_feat, n_feat, kernel_size=3, stride=1, padding=1, bias=bias, groups=groups)
        )

        self.act = act
        
        self.gcnet = ContextBlock(n_feat, bias=bias)

    def forward(self, x):
        res = self.body(x)
        res = self.act(self.gcnet(res))
        res += x
        return res


##########################################################################
##---------- Resizing Modules ----------    
class Down(nn.Module):
    def __init__(self, in_channels, chan_factor, bias=False):
        super(Down, self).__init__()

        self.bot = nn.Sequential(
            nn.AvgPool2d(2, ceil_mode=True, count_include_pad=False),
            nn.Conv2d(in_channels, int(in_channels*chan_factor), 1, stride=1, padding=0, bias=bias)
            )

    def forward(self, x):
        return self.bot(x)

class DownSample(nn.Module):
    def __init__(self, in_channels, scale_factor, chan_factor=2, kernel_size=3):
        super(DownSample, self).__init__()
        self.scale_factor = int(np.log2(scale_factor))

        modules_body = []
        for i in range(self.scale_factor):
            modules_body.append(Down(in_channels, chan_factor))
            in_channels = int(in_channels * chan_factor)
        
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        x = self.body(x)
        return x

#########-----上采样--------########
class Up(nn.Module):
    def __init__(self, in_channels, chan_factor, bias=False):
        super(Up, self).__init__()

        self.bot = nn.Sequential(
            nn.Conv2d(in_channels, int(in_channels//chan_factor), 1, stride=1, padding=0, bias=bias),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=bias)
            )

    def forward(self, x):
        return self.bot(x)

#########-----下采样--------########
class UpSample(nn.Module):
    def __init__(self, in_channels, scale_factor, chan_factor=2, kernel_size=3):
        super(UpSample, self).__init__()
        self.scale_factor = int(np.log2(scale_factor))

        modules_body = []
        for i in range(self.scale_factor):
            modules_body.append(Up(in_channels, chan_factor))
            in_channels = int(in_channels // chan_factor)
        
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        x = self.body(x)
        return x


##########################################################################
##---------- 对应网络的模块A (n_Ablock) ----------
class Ablock(nn.Module):
    def __init__(self, n_feat, height, width, chan_factor, bias,groups):
        super(Ablock, self).__init__()

        self.n_feat, self.height, self.width = n_feat, height, width

        self.dau_top = FEB(int(n_feat*chan_factor**0), bias=bias, groups=groups)
        self.dau_mid = FEB(int(n_feat*chan_factor**1), bias=bias, groups=groups)
        self.dau_bot = FEB(int(n_feat*chan_factor**2), bias=bias, groups=groups)

        self.down2 = DownSample(int((chan_factor**0)*n_feat),2,chan_factor)
        self.down4 = nn.Sequential(
            DownSample(int((chan_factor**0)*n_feat),2,chan_factor), 
            DownSample(int((chan_factor**1)*n_feat),2,chan_factor)
        )

        self.up21_1 = UpSample(int((chan_factor**1)*n_feat),2,chan_factor)
        self.up21_2 = UpSample(int((chan_factor**1)*n_feat),2,chan_factor)
        self.up32_1 = UpSample(int((chan_factor**2)*n_feat),2,chan_factor)
        self.up32_2 = UpSample(int((chan_factor**2)*n_feat),2,chan_factor)

        self.conv_out = nn.Conv2d(n_feat, n_feat, kernel_size=1, padding=0, bias=bias)

        # only two inputs for FIB
        self.FIB_top = FIB(int(n_feat*chan_factor**0), 2)
        self.FIB_mid = FIB(int(n_feat*chan_factor**1), 2)

    def forward(self, x):
        x_top = x.clone()
        x_mid = self.down2(x_top)
        x_bot = self.down4(x_top)

        x_top = self.dau_top(x_top)
        x_mid = self.dau_mid(x_mid)
        x_bot = self.dau_bot(x_bot)

        x_mid = self.FIB_mid([x_mid, self.up32_1(x_bot)])  # 输入两个特征
        x_top = self.FIB_top([x_top, self.up21_1(x_mid)])

        x_top = self.dau_top(x_top)
        x_mid = self.dau_mid(x_mid)
        x_bot = self.dau_bot(x_bot)

        x_mid = self.FIB_mid([x_mid, self.up32_2(x_bot)])
        x_top = self.FIB_top([x_top, self.up21_2(x_mid)])

        out = self.conv_out(x_top)
        out = out + x

        return out

##########################################################################
##---------- 残差组 ----------
class RG(nn.Module):
    def __init__(self, n_feat, n_Ablock, height, width, chan_factor, bias=False, groups=1):
        super(RG, self).__init__()
        modules_body = [Ablock(n_feat, height, width, chan_factor, bias, groups) for _ in range(n_Ablock)]
        modules_body.append(nn.Conv2d(n_feat, n_feat, kernel_size=3, stride=1, padding=1, bias=bias))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


##########################################################################
##---------- W_model---主干  -----------------------
class W_model(nn.Module):
    def __init__(self,
        inp_channels=3,
        out_channels=3,
        n_feat=80,
        chan_factor=1.5,
        n_RG=4,
        n_Ablock=2,
        height=3,
        width=2,
        scale=1,
        bias=False,
        task= None
    ):
        super(W_model, self).__init__()

        kernel_size=3
        self.task = task

        self.conv_in = nn.Conv2d(inp_channels, n_feat, kernel_size=3, padding=1, bias=bias)

        modules_body = []
        
        modules_body.append(RG(n_feat, n_Ablock, height, width, chan_factor, bias, groups=1))  # 中间块
        modules_body.append(RG(n_feat, n_Ablock, height, width, chan_factor, bias, groups=2))  # 中间块
        modules_body.append(RG(n_feat, n_Ablock, height, width, chan_factor, bias, groups=4))  # 中间块
        modules_body.append(RG(n_feat, n_Ablock, height, width, chan_factor, bias, groups=4))  # 中间块

        self.body = nn.Sequential(*modules_body)
        self.conv_out = nn.Conv2d(n_feat, out_channels, kernel_size=3, padding=1, bias=bias)
        

    def forward(self, inp_img):
        shallow_feats = self.conv_in(inp_img)
        deep_feats = self.body(shallow_feats)

        if self.task == 'water':
            deep_feats += shallow_feats
            out_img = self.conv_out(deep_feats)

        else:
            out_img = self.conv_out(deep_feats)
            out_img += inp_img

        return out_img
net=W_model()
img = torch.rand((1,3,224,224))
torch.onnx.export(model=net,args=img,f='model.onnx', input_names=[ 'image'], output_names=[ 'feature_map'],opset_version=11)
onnx.save(onnx.shape_inference.infer_shapes(onnx.load("model.onnx")),"model.onnx")
netron.start("model.onnx")