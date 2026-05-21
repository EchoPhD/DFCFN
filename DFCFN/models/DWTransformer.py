import torch
import torch.nn.functional as F
from torch import nn
from scipy.io import savemat
from torchvision import models
from einops import rearrange
import numbers
from torchvision import transforms
from scipy.io import loadmat, savemat
from DWT_IDWT.DWT_IDWT_layer import DWT_1D, DWT_2D, IDWT_1D, IDWT_2D

import numpy as np
import torch
import math
from torch.nn import Module, Sequential, Conv2d, ReLU,AdaptiveMaxPool2d, AdaptiveAvgPool2d, \
    NLLLoss, BCELoss, CrossEntropyLoss, AvgPool2d, MaxPool2d, Parameter, Linear, Sigmoid, Softmax, Dropout, Embedding
from torch.nn import functional as F
from torch.autograd import Variable
#torch_ver = torch.__version__[:3]

import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange

'''------------------------------------------------
conv1x1函数： 定义1×1卷积层
输入：输入通道数、输出通道数
输出：设置好的卷积层
------------------------------------------------'''
def conv1x1(in_channels, out_channels, stride=1):
    #return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=True)
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False)

'''------------------------------------------------
conv3x3函数： 定义3×3卷积层
输入：输入通道数、输出通道数
输出：设置好的卷积层
------------------------------------------------'''
def conv3x3(in_channels, out_channels, stride=1):
    #return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=True)
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)

'''------------------------------------------------
ResBlock函数： 定义残差块网络
输入：输入通道数、输出通道数
输出：两层3×3卷积、ReLU激活的残差块网络
------------------------------------------------'''
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, res_scale=1):
        super(ResBlock, self).__init__()
        self.res_scale = res_scale
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)

    def forward(self, x):
        x1 = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = out * self.res_scale + x1
        return out
    
    
'''========================================================================
Feature_extraction模块： 浅特征提取，卷积+残差+卷积 用于通道调整
输入：原始尺寸的HSI/pan
输出：256通道
========================================================================'''
class Feature_extraction(nn.Module):
    def __init__(self, in_feats, num_res_blocks, n_feats, res_scale):
        super(Feature_extraction, self).__init__()
        self.num_res_blocks = num_res_blocks
        self.conv_head = conv3x3(in_feats, n_feats)
        
        self.RBs = nn.ModuleList()
        for i in range(self.num_res_blocks):
            self.RBs.append(ResBlock(in_channels=n_feats, out_channels=n_feats, res_scale=res_scale))
        self.conv_tail = conv3x3(n_feats, n_feats)
        
        # 批归一化和层归一化
        self.OutBN = nn.BatchNorm2d(num_features=n_feats)  
        
    def forward(self, x):
        x = F.relu(self.conv_head(x))
        x1 = x
        for i in range(self.num_res_blocks):
            x = self.RBs[i](x)
        x = self.conv_tail(x)
        #x = x + x1
        
        #x = self.OutBN(x)
        #x = F.relu(x)
        return x

'''========================================================================
Feature_adjustment模块：特征调整模块，卷积+残差+卷积
输入：原始尺寸的HSI
输出：HSI特征图
========================================================================'''
class Feature_adjustment(nn.Module):
    def __init__(self, in_feats, num_res_blocks, n_feats, res_scale):
        super(Feature_adjustment, self).__init__()
        self.num_res_blocks = num_res_blocks
        self.conv_head = conv3x3(in_feats, n_feats)
        
        self.RBs = nn.ModuleList()
        for i in range(self.num_res_blocks):
            self.RBs.append(ResBlock(in_channels=n_feats, out_channels=n_feats, 
                res_scale=res_scale))
        self.conv_tail = conv3x3(n_feats, n_feats)
        
    def forward(self, x):
        x = F.relu(self.conv_head(x))
        x1 = x
        for i in range(self.num_res_blocks):
            x = self.RBs[i](x)
        #x = F.relu(self.conv_tail(x))
        x = self.conv_tail(x)
        #x = x + x1
        return x


'''===========================================================================
晶格模块中PAN的空间注意力模块(a part of CBAM)：
输入：通道调整后的PAN
输出：空间增强后的权重A
===========================================================================''' 
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)      #self.sigmoid(x) * x

'''===========================================================================
晶格模块中HSI的通道注意力模块(a part of CBAM)：
输入：通道调整后的HSI
输出：通道增强后权重B
===========================================================================''' 
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        # 共享权重的MLP
        self.fc1   = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)        #self.sigmoid(out) * x

'''========================================================================
晶格模块： 用于替换原始残差第一阶段融合HSI与PAN
输入：原始尺寸的HSI/pan
输出：256通道融合结果
========================================================================'''
class LatticeBlock(nn.Module):
    def __init__(self, nFeat):
        super(LatticeBlock, self).__init__()
        ''' 此版本是两阶段的晶格结构，第一阶段是混合的，PAN和HSI两个分支混合，第二阶段是在第一阶段最后线性层输出的结果上进行单图处理 '''
        self.n_feats = nFeat    #PAN和HSI输入特征层数

        ''' ================================ 第 1 阶段 ===================================== '''
        #PAN支路的RBs块
        self.conv_head = conv3x3(1, self.n_feats)
        self.conv_block0 = nn.ModuleList()
        for i in range(2):
            self.conv_block0.append(ResBlock(in_channels=self.n_feats, out_channels=self.n_feats, res_scale=1))
        self.pan_conv_tail = conv3x3(self.n_feats, self.n_feats)

        '''注：这两块的输出结果是权重'''
        self.PAN_A_i_1 = SpatialAttention(kernel_size=7)
        self.HSI_B_i_1 = ChannelAttention(nFeat, ratio=8)

        #HSI支路的RBs块
        self.conv_block1 = nn.ModuleList()
        for i in range(2):
            self.conv_block1.append(ResBlock(in_channels=self.n_feats, out_channels=self.n_feats, res_scale=1))
        self.hsi_conv_tail = conv3x3(self.n_feats, self.n_feats)

        '''注：这两块的输出结果是权重'''
        self.PAN_A_i = SpatialAttention(kernel_size=7)
        self.HSI_B_i = ChannelAttention(nFeat, ratio=8)
        
        self.compress = nn.Conv2d(2 * nFeat, nFeat, kernel_size=1, padding=0, bias=True)
        
    #def forward(self, x):
    def forward(self, pan, hsi):
        
        ''' ==================================== 第 1 阶段 ========================================= '''
        ''' analyse unit '''
        pan = F.relu(self.conv_head(pan))   #pan图调整通道到256
        pan_feature_i_1 = pan
        #pan图经过残差块
        for i in range(2):
            pan_feature_i_1 = self.conv_block0[i](pan_feature_i_1)
        pan_feature_i_1 = self.pan_conv_tail(pan_feature_i_1)
        
        pan_feature_A_i_1 = self.PAN_A_i_1(pan_feature_i_1) #第一个晶格结构PAN图权重A_i-1
        hsi_feature_B_i_1 = self.HSI_B_i_1(hsi) #第一个晶格结构HSI图权重B_i-1        
        ##开始相加
        hsi_next = hsi + pan_feature_A_i_1*pan_feature_i_1  #HSI + PAN权重A_i-1 * PAN
        pan_next = pan_feature_i_1 + hsi_feature_B_i_1* hsi  #PAN + HSI权重B_i-1 * HSI
        

        ''' synthes unit '''
        hsi_feature_i = hsi_next
        #HSI图经过残差块
        for i in range(2):
            hsi_feature_i = self.conv_block1[i](hsi_feature_i)
        hsi_feature_i = self.hsi_conv_tail(hsi_feature_i)
        
        hsi_feature_B_i = self.HSI_B_i(hsi_feature_i)     #第二个晶格结构HSI图权重B_i   
        pan_feature_A_i = self.PAN_A_i(pan_next)  #第二个晶格结构PAN图权重A_i
        
        hsi_finial = hsi_feature_i + pan_feature_A_i*pan_next
        pan_finial = pan_next + hsi_feature_B_i*hsi_feature_i      

        out = torch.cat((hsi_finial, pan_finial), 1)
        out = self.compress(out)
                      
        return out

'''========================================================================
DWT_data函数： 小波变换/逆变换
输入：[batch_size, channels, image_size,image_size]
输出：data_ave, data_hor, data_ver, data_dia （低频均值分量、水平高频lh、垂直高频hl、对角高频hh）
========================================================================'''
def DWT_data(self, data, I_NO):
    ''' I_NO=0, 则进行小波变换，否则进行小波逆变换'''
    if not I_NO:
        # 2位小波变换 一种比较简单的小波 haar(哈尔小波）
        dwt = DWT_2D("haar")
        #xll, xlh, xhl, xhh = dwt(data)
        final_result = dwt(data)
    else:
        iwt = IDWT_2D("haar")
        final_result = iwt(data[0],data[1],data[2],data[3])

    #####以下内容为保存小波变换的结果为mat文件
    # filepath = "./temp_data/"
    ####tensor转numpy
    # xll = xll.cpu().detach().numpy()
    # xlh = xlh.cpu().detach().numpy()
    # xhl = xhl.cpu().detach().numpy()
    # xhh = xhh.cpu().detach().numpy()
    # dwt_data = dwt_data.cpu().detach().numpy()
    # savemat("xll.mat", {'xll':xll})
    # savemat("xlh.mat", {'xlh':xlh})
    # savemat("xhl.mat", {'xhl':xhl})
    # savemat("xhh.mat", {'xhh':xhh})
    # savemat("dwt_data.mat", {'dwt_data': dwt_data})
    # return [xll,xlh,xhl,xhh]
    return final_result



'''================================================
attention模块：self attention或者cross attention模块。 
输入：
输出：
================================================'''
class Attention(nn.Module):
    ''' Scaled Dot-Product Attention 缩放点积注意'''
    def __init__(self, channels,num_heads):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.softmax = nn.Softmax(dim=-1)
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.norm_q_01 = nn.BatchNorm2d(num_features = self.channels)   #第一层Q支路LN归一化层
        self.norm_k_01 = nn.BatchNorm2d(num_features = self.channels)   #第一层K支路LN归一化层
        self.norm_v_01 = nn.BatchNorm2d(num_features = self.channels)   #第一层V支路LN归一化层
        
        '''第一层Q支路中的 1×1 + 3×3 卷积层'''
        self.Q_01 = nn.Sequential(
            nn.Conv2d(in_channels=self.channels, out_channels=self.channels, kernel_size=1, stride=1, bias=False),
            nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, groups=self.channels, bias=False))        

        '''第一层K支路中的 1×1 + 3×3 卷积层'''
        self.K_01 = nn.Sequential(
            nn.Conv2d(in_channels=self.channels, out_channels=self.channels, kernel_size=1, stride=1, bias=False),
            nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, groups=self.channels, bias=False))
        
        '''第一层V支路中的 1×1 + 3×3 卷积层'''
        self.V_01 = nn.Sequential(
            nn.Conv2d(in_channels=self.channels, out_channels=self.channels, kernel_size=1, stride=1, bias=False),
            nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, groups=self.channels, bias=False))   
        
        # 批归一化和层归一化
        self.OutBN = nn.BatchNorm2d(num_features = self.channels)              
        self.project_out = nn.Conv2d(in_channels=self.channels, out_channels=self.channels, kernel_size=1, stride=1, bias=False)        
    
    def forward(self, v_01, k_01, q_01):
        output_01 = v_01    #加入DHP
        
        ''' 归一化层 '''
        q_01 = self.norm_q_01(q_01)   #第一层Q支路LN归一化层
        k_01 = self.norm_k_01(k_01)   #第一层K支路LN归一化层
        v_01 = self.norm_v_01(v_01)   #第一层V支路LN归一化层        

        ''' 1*1 + 3*3 卷积获取通道间上下文 '''
        q_01 = self.Q_01(q_01)
        k_01 = self.K_01(k_01)
        v_01 = self.V_01(v_01)

        '''重塑K,Q和V...'''                
        b, c, h, w = q_01.size(0), q_01.size(1), q_01.size(2), q_01.size(3)
        #q_01 = q_01.view(b, c, h*w)
        #k_01 = k_01.view(b, c, h*w)
        #v_01 = v_01.view(b, c, h*w)

        q_01 = rearrange(q_01, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k_01 = rearrange(k_01, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v_01 = rearrange(v_01, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        '''自注意力注意力计算'''
        #output_01 = v_01    
        q_01 = torch.nn.functional.normalize(q_01, dim=-1)
        k_01 = torch.nn.functional.normalize(k_01, dim=-1)
       
        # 计算注意力      
        attn_01 = (q_01 @ k_01.transpose(-2, -1)) * self.temperature        
        
        #标准化(SoftMax)
        attn_01    = F.softmax(attn_01, dim=-1)

        out = (attn_01 @ v_01)
        
        out = output_01 + rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        output = self.project_out(out)   
        
        output = F.relu(self.OutBN(output))
        
        return output

'''================================================
Cross_Attention模块：cross attention模块。 
输入：
输出：
================================================'''
class Cross_Attention(nn.Module):
    ''' Scaled Dot-Product Attention 缩放点积注意'''
    def __init__(self, channels,num_heads):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.softmax = nn.Softmax(dim=-1)
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        #self.fliper = transforms.RandomHorizontalFlip(1)        
             
        self.norm_q_01 = nn.BatchNorm2d(num_features = self.channels)   #第一层Q支路LN归一化层
        self.norm_k_01 = nn.BatchNorm2d(num_features = self.channels*2)   #第一层K支路LN归一化层
        self.norm_v_01 = nn.BatchNorm2d(num_features = self.channels*2)   #第一层V支路LN归一化层

        '''第一层Q支路中的 1×1 + 3×3 卷积层'''
        self.Q_01 = nn.Sequential(
            nn.Conv2d(in_channels=self.channels, out_channels=self.channels, kernel_size=1, stride=1, bias=False),
            nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, groups=self.channels, bias=False))        

        '''第一层K支路中的 1×1 + 3×3 卷积层'''
        self.K_01 = nn.Sequential(
            nn.Conv2d(in_channels=self.channels*2, out_channels=self.channels, kernel_size=1, stride=1, bias=False),
            nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, groups=self.channels, bias=False))
        
        '''第一层V支路中的 1×1 + 3×3 卷积层'''
        self.V_01 = nn.Sequential(
            nn.Conv2d(in_channels=self.channels*2, out_channels=self.channels, kernel_size=1, stride=1, bias=False),
            nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, groups=self.channels, bias=False))   
        
        # 批归一化和层归一化
        self.OutBN = nn.BatchNorm2d(num_features = self.channels)              
        self.project_out = nn.Conv2d(in_channels=self.channels, out_channels=self.channels, kernel_size=1, stride=1, bias=False)        
    
    def forward(self, v_01, k_01, q_01):
        output_01 = v_01    
        
        ''' 归一化层 '''
        q_01 = self.norm_q_01(q_01)   #第一层Q支路LN归一化层
        
        kv_01= torch.cat((q_01,k_01), dim=1)  
        k_01, v_01 = kv_01,kv_01
        k_01 = self.norm_k_01(k_01)   #第一层K支路LN归一化层
        v_01 = self.norm_v_01(v_01)   #第一层V支路LN归一化层        

        ''' 1*1 + 3*3 卷积获取通道间上下文 '''      
        q_01 = self.Q_01(q_01)        
        k_01 = self.K_01(k_01)
        v_01 = self.V_01(v_01)

        '''重塑K,Q和V...'''                
        b, c, h, w = q_01.size(0), q_01.size(1), q_01.size(2), q_01.size(3)

        q_01 = rearrange(q_01, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k_01 = rearrange(k_01, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v_01 = rearrange(v_01, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        '''自注意力注意力计算'''
        #output_01 = v_01    
        q_01 = torch.nn.functional.normalize(q_01, dim=-1)
        k_01 = torch.nn.functional.normalize(k_01, dim=-1)
       
        # 计算注意力      
        attn_01 = (q_01 @ k_01.transpose(-2, -1)) * self.temperature        
        
        #标准化(SoftMax)
        attn_01    = F.softmax(attn_01, dim=-1)

        out = (attn_01 @ v_01)
        
        out = output_01 + rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        output = self.project_out(out)   
        
        #output = F.relu(self.OutBN(output))
        output = self.OutBN(output)
        
        return output

'''================================================
Conditional_entropy模块：条件熵计算
输入：
输出：
================================================
def Conditional_entropy(self, feature_map_A, feature_map_B):
    # 假设有两个特征图 A 和 B，它们的形状为 [B, C, H, W]
    # 这里 B 是批次大小，C 是通道数，H 是高度，W 是宽度

    # 计算 A 在给定 B 的条件下的概率分布
    prob_A_given_B = F.softmax(feature_map_A, dim=1)  # 在通道维度上进行归一化

    # 计算条件熵 H(A|B)
    log_prob_A_given_B = torch.log(prob_A_given_B + 1e-10)  # 避免取对数时出现负无穷
    conditional_entropy_A_given_B = -torch.sum(prob_A_given_B * log_prob_A_given_B, dim=1).mean()

    # 计算 B 在给定 A 的条件下的概率分布
    prob_B_given_A = F.softmax(feature_map_B, dim=1)  # 在通道维度上进行归一化

    # 计算条件熵 H(B|A)
    log_prob_B_given_A = torch.log(prob_B_given_A + 1e-10)  # 避免取对数时出现负无穷
    conditional_entropy_B_given_A = -torch.sum(prob_B_given_A * log_prob_B_given_A, dim=1).mean()

    print("条件熵 H(A|B):", conditional_entropy_A_given_B.item())
    print("条件熵 H(B|A):", conditional_entropy_B_given_A.item())
    
    output = { "CE_A_B": conditional_entropy_A_given_B.item(), "CE_B_A": conditional_entropy_B_given_A.item()}
    return output
'''

'''================================================
DWTransformer模块：总启动程序 
输入：
输出：
================================================'''
class DWTransformer(nn.Module):
    def __init__(self, config):
        super(DWTransformer, self).__init__()
        self.is_DHP_MS      = config["is_DHP_MS"]
        self.num_head      = config["N_modules"]
        self.in_channels    = config[config["train_dataset"]]["spectral_bands"] #光谱通道数
        self.out_channels   = config[config["train_dataset"]]["spectral_bands"]
        self.factor         = config[config["train_dataset"]]["factor"]     #尺寸缩放比例

        self.num_res_blocks = [2, 2] #Feature_extraction模块与Feature_adjustment模块中所用的残差块数量
        #self.n_feats        = 256
        self.res_scale      = 1 #残差缩放，默认为1 
        self.feature_chanels = [256, 768]
        
        ''' LR-HSI和PAN的特征通道调整(Feature_extraction模块) '''
        self.lrhsi_feature_extraction = Feature_extraction(self.in_channels, self.num_res_blocks[0], self.feature_chanels[0], self.res_scale)
        self.up_lrhsi_feature_extraction = Feature_extraction(self.in_channels, self.num_res_blocks[0], self.feature_chanels[0], self.res_scale)          
        #self.pan_feature_extraction = Feature_extraction(1, self.num_res_blocks[0], self.feature_chanels[0], self.res_scale) 
        
        ''' 晶格块 '''
        self.first_fusion_block = LatticeBlock(self.feature_chanels[0])
        #self.SCIN = SPIN_Block(dim = self.feature_chanels[0], stoken_size = [1,16], heads = self.num_head, mlp_dim = self.feature_chanels[0])
        
        ''' LR HSI与 PAN 注意力特征融合, 使用交叉注意力,高频部分可以分三个计算，也可以先cat，后使用一个计算，这里先采用分开计算'''
        self.low_cross_attention    = Cross_Attention( channels = self.feature_chanels[0], num_heads = self.num_head)
        self.high_hor_cross_attention   = Cross_Attention( channels = self.feature_chanels[0], num_heads = self.num_head)  #水平
        self.high_ver_cross_attention   = Cross_Attention( channels = self.feature_chanels[0], num_heads = self.num_head)  #垂直
        self.high_dia_cross_attention   = Cross_Attention( channels = self.feature_chanels[0], num_heads = self.num_head)  #对角
        
        '''逆小波变换的两层注意力特征融合，使用交叉注意力'''
        self.cross_attention_02 = Cross_Attention( channels = self.feature_chanels[0], num_heads = self.num_head)
        self.cross_attention_01 = Cross_Attention( channels = self.feature_chanels[0], num_heads = self.num_head)
        
        ''' 最终融合结果特征调整(Feature_adjustment模块) '''
        self.feature_adjustment = Feature_adjustment(self.feature_chanels[0], self.num_res_blocks[0], self.out_channels, self.res_scale)  
        
        ''' 批处理归一化 '''
        self.BN_D3U1 = nn.BatchNorm2d(self.feature_chanels[1])
    
    ''' 小波变换计算函数 '''
    def DWT_data(self, data, I_NO):
        ''' I_NO=0, 则进行小波变换，否则进行小波逆变换'''
        if not I_NO:
            # 2位小波变换 一种比较简单的小波 haar(哈尔小波）
            dwt = DWT_2D("haar")
            #xll, xlh, xhl, xhh = dwt(data)
            final_result = dwt(data)
        else:
            iwt = IDWT_2D("haar")
            final_result = iwt(data[0],data[1],data[2],data[3])

        return final_result

    ''' 条件熵计算函数 '''
    def Conditional_entropy(self, feature_map_A, feature_map_B):
        # 假设有两个特征图 A 和 B，它们的形状为 [B, C, H, W]
        # 这里 B 是批次大小，C 是通道数，H 是高度，W 是宽度

        # 计算 A 在给定 B 的条件下的概率分布
        #prob_A_given_B = F.softmax(feature_map_A, dim=1)  # 在通道维度上进行归一化
        prob_A_given_B = F.softmax(feature_map_B, dim=1)  # 在通道维度上进行归一化

        # 计算条件熵 H(A|B)
        log_prob_A_given_B = torch.log(prob_A_given_B + 1e-10)  # 避免取对数时出现负无穷
        conditional_entropy_A_given_B = -torch.sum(prob_A_given_B * log_prob_A_given_B, dim=1).mean()

        # 计算 B 在给定 A 的条件下的概率分布
        #prob_B_given_A = F.softmax(feature_map_B, dim=1)  # 在通道维度上进行归一化
        prob_B_given_A = F.softmax(feature_map_A, dim=1)  # 在通道维度上进行归一化

        # 计算条件熵 H(B|A)
        log_prob_B_given_A = torch.log(prob_B_given_A + 1e-10)  # 避免取对数时出现负无穷
        conditional_entropy_B_given_A = -torch.sum(prob_B_given_A * log_prob_B_given_A, dim=1).mean()

        #CE_A_B_value = conditional_entropy_A_given_B.item() / (conditional_entropy_A_given_B.item() + conditional_entropy_B_given_A.item())
        #CE_B_A_value = conditional_entropy_B_given_A.item() / (conditional_entropy_A_given_B.item() + conditional_entropy_B_given_A.item())

        CE_A_B_value = conditional_entropy_A_given_B.item()
        CE_B_A_value = conditional_entropy_B_given_A.item()

        print("条件熵 H(A|B):-------------------->", CE_A_B_value)
        print("条件熵 H(B|A):-------------------->", CE_B_A_value)
    
        output = { "CE_A_B": CE_A_B_value, "CE_B_A": CE_B_A_value}
        return output 
        
        
    def forward(self, LR_HSI, PAN, HSI_DHP):
        
        #调整PAN尺寸[batch_size, 通道数，HR_size, HR_size]
        PAN   = PAN.unsqueeze(dim=1)
        
        ''' 初始特征通道调整 '''
        F_lrhsi = self.lrhsi_feature_extraction(LR_HSI)
        #F_pan = self.pan_feature_extraction(PAN)
        UP_LR_HSI = F.interpolate(LR_HSI, scale_factor=(self.factor,self.factor),mode ='bicubic')   #上采样的LR-HSI，用于晶格结构
        F_up_lrhsi = self.up_lrhsi_feature_extraction(UP_LR_HSI)    #HSI调整通道
        #F_pan = PAN #PAN图不需要调整，在晶格中会对PAN进行调整
        F_pan = self.first_fusion_block(PAN, F_up_lrhsi)
        
        ''' PAN图小波变换（三层） '''
        pan_dwt_01 = self.DWT_data(F_pan, 0)
        pan_dwt_01_ave, pan_dwt_01_hor, pan_dwt_01_ver, pan_dwt_01_dia = pan_dwt_01[0], pan_dwt_01[1], pan_dwt_01[2], pan_dwt_01[3]
        
        pan_dwt_02 = self.DWT_data(pan_dwt_01_ave, 0)
        pan_dwt_02_ave, pan_dwt_02_hor, pan_dwt_02_ver, pan_dwt_02_dia = pan_dwt_02[0], pan_dwt_02[1], pan_dwt_02[2], pan_dwt_02[3]        
        
        pan_dwt_03 = self.DWT_data(pan_dwt_02_ave, 0)
        pan_dwt_03_ave, pan_dwt_03_hor, pan_dwt_03_ver, pan_dwt_03_dia = pan_dwt_03[0], pan_dwt_03[1], pan_dwt_03[2], pan_dwt_03[3]        
        
        ''' LR-HSI图小波变换（一层，但是为了标识方便，这里使用03标记） '''
        lrhsi_dwt_03 = self.DWT_data(F_lrhsi, 0)
        lrhsi_dwt_03_ave, lrhsi_dwt_03_hor, lrhsi_dwt_03_ver, lrhsi_dwt_03_dia = lrhsi_dwt_03[0], lrhsi_dwt_03[1], lrhsi_dwt_03[2], lrhsi_dwt_03[3]       
        
        
        ''' PAN与LR-HSI在小波域上的首次融合，低频对低频，高频对高频 (V,K,Q)'''
        CE_03_ave = self.Conditional_entropy(lrhsi_dwt_03_ave, pan_dwt_03_ave)  ### 条件熵计算 Conditional_entropy(A, B)  CE = {CE_A_B, CE_B_A}
        lrhsi_dwt_03_ave = CE_03_ave["CE_A_B"] * lrhsi_dwt_03_ave + CE_03_ave["CE_B_A"] * pan_dwt_03_ave  ### 数据融合
        #lrhsi_dwt_03_ave = CE_03_ave["CE_B_A"] * lrhsi_dwt_03_ave + CE_03_ave["CE_A_B"] * pan_dwt_03_ave  ### 数据融合
        HSI_low_03 = self.low_cross_attention(lrhsi_dwt_03_ave, lrhsi_dwt_03_ave, pan_dwt_03_ave)   # 低频交叉注意力，HSI的低频做KV，PAN图做Q

        # pan_high_f = torch.cat((LR_HSI_UP, PAN), dim=1)
        CE_03_hor = self.Conditional_entropy(lrhsi_dwt_03_hor, pan_dwt_03_hor)  ### 条件熵计算 Conditional_entropy(A, B)  CE = {CE_A_B, CE_B_A}
        lrhsi_dwt_03_hor = CE_03_hor["CE_A_B"] * lrhsi_dwt_03_hor + CE_03_hor["CE_B_A"] * pan_dwt_03_hor  ### 数据融合
        #lrhsi_dwt_03_hor = CE_03_hor["CE_B_A"] * lrhsi_dwt_03_hor + CE_03_hor["CE_A_B"] * pan_dwt_03_hor  ### 数据融合
        
        CE_03_ver = self.Conditional_entropy(lrhsi_dwt_03_ver, pan_dwt_03_ver)  ### 条件熵计算 Conditional_entropy(A, B)  CE = {CE_A_B, CE_B_A}
        lrhsi_dwt_03_ver = CE_03_ver["CE_A_B"] * lrhsi_dwt_03_ver + CE_03_ver["CE_B_A"] * pan_dwt_03_ver  ### 数据融合
        #lrhsi_dwt_03_ver = CE_03_ver["CE_B_A"] * lrhsi_dwt_03_ver + CE_03_ver["CE_A_B"] * pan_dwt_03_ver  ### 数据融合
        
        CE_03_dia = self.Conditional_entropy(lrhsi_dwt_03_dia, pan_dwt_03_dia)  ### 条件熵计算 Conditional_entropy(A, B)  CE = {CE_A_B, CE_B_A}
        lrhsi_dwt_03_dia = CE_03_dia["CE_A_B"] * lrhsi_dwt_03_dia + CE_03_dia["CE_B_A"] * pan_dwt_03_dia  ### 数据融合
        #lrhsi_dwt_03_dia = CE_03_dia["CE_B_A"] * lrhsi_dwt_03_dia + CE_03_dia["CE_A_B"] * pan_dwt_03_dia  ### 数据融合
        
        HSI_high_hor_03 = self.high_hor_cross_attention(lrhsi_dwt_03_hor, lrhsi_dwt_03_hor, pan_dwt_03_hor)   # 水平高频交叉注意力，HSI的高频做KV，PAN图做Q
        HSI_high_ver_03 = self.high_ver_cross_attention(lrhsi_dwt_03_ver, lrhsi_dwt_03_ver, pan_dwt_03_ver)   # 垂直高频交叉注意力，HSI的高频做KV，PAN图做Q
        HSI_high_dia_03 = self.high_dia_cross_attention(lrhsi_dwt_03_dia, lrhsi_dwt_03_dia, pan_dwt_03_dia)   # 对角高频交叉注意力，HSI的高频做KV，PAN图做Q
        
        '''第一次逆小波变换'''
        IDWT_data = [HSI_low_03, HSI_high_hor_03, HSI_high_ver_03, HSI_high_dia_03]
        hsi_dwt_03_ave = self.DWT_data(IDWT_data, 1)    # H/4 × W/4
        
        ''' 第二次跨域低频交叉注意力融合 (V,K,Q)'''
        CE_03_hsi_ave = self.Conditional_entropy(hsi_dwt_03_ave, pan_dwt_02_ave)  ### 条件熵计算 Conditional_entropy(A, B)  CE = {CE_A_B, CE_B_A}
        hsi_dwt_03_ave = CE_03_hsi_ave["CE_A_B"] * hsi_dwt_03_ave + CE_03_hsi_ave["CE_B_A"] * pan_dwt_02_ave  ### 数据融合
        #hsi_dwt_03_ave = CE_03_hsi_ave["CE_B_A"] * hsi_dwt_03_ave + CE_03_hsi_ave["CE_A_B"] * pan_dwt_02_ave  ### 数据融合
        
        hsi_dwt_02_ave = self.cross_attention_02(hsi_dwt_03_ave, hsi_dwt_03_ave, pan_dwt_02_ave)


        ''' 第二次逆小波变换 '''
        IDWT_data = [hsi_dwt_02_ave, pan_dwt_02_hor, pan_dwt_02_ver, pan_dwt_02_dia]
        hsi_dwt_02_ave = self.DWT_data(IDWT_data, 1)    # H/2 × W/2

        ''' 第三次跨域低频交叉注意力融合 (V,K,Q) '''     
        CE_02_hsi_ave = self.Conditional_entropy(hsi_dwt_02_ave, pan_dwt_01_ave)  ### 条件熵计算 Conditional_entropy(A, B)  CE = {CE_A_B, CE_B_A}
        hsi_dwt_02_ave = CE_02_hsi_ave["CE_A_B"] * hsi_dwt_02_ave + CE_02_hsi_ave["CE_B_A"] * pan_dwt_01_ave  ### 数据融合
        #hsi_dwt_02_ave = CE_02_hsi_ave["CE_B_A"] * hsi_dwt_02_ave + CE_02_hsi_ave["CE_A_B"] * pan_dwt_01_ave  ### 数据融合
        
        hsi_dwt_01_ave = self.cross_attention_01(hsi_dwt_02_ave, hsi_dwt_02_ave, pan_dwt_01_ave)

        ''' 第三次逆小波变换 '''
        IDWT_data = [hsi_dwt_01_ave, pan_dwt_01_hor, pan_dwt_01_ver, pan_dwt_01_dia]
        hsi_dwt_01 = self.DWT_data(IDWT_data, 1)    # H × W
        
        ''' 特征调整，输出最终结果 '''
        HR_HSI = self.feature_adjustment(hsi_dwt_01)

        output = { "pred": HR_HSI}
        return output


