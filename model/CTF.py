from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable

# class DepthwiseSeparableConv(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(DepthwiseSeparableConv, self).__init__()
#         self.depthwise = nn.Conv1d(in_channels, in_channels, kernel_size=1, groups=in_channels)  # 深度卷积
#         self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1)  # 逐点卷积

#     def forward(self, x):
#         return self.pointwise(self.depthwise(x))

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DepthwiseSeparableConv, self).__init__()
        
        self.depthwise_3x3 = nn.Conv1d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
        
        self.depthwise_5x5 = nn.Conv1d(in_channels, in_channels, kernel_size=5, padding=2, groups=in_channels)

        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        
        self.elu = nn.ELU()

    def forward(self, x):
        x_3x3 = self.depthwise_3x3(x)
        x_5x5 = self.depthwise_5x5(x)
        x = x_3x3 + x_5x5
        x = self.pointwise(x)
        
        x = self.elu(x)
        return x
    
class STN3d(nn.Module):
    def __init__(self):
        super(STN3d, self).__init__()
        self.conv1 = DepthwiseSeparableConv(3, 64)
        self.conv2 = DepthwiseSeparableConv(64, 128)
        self.conv3 = DepthwiseSeparableConv(128, 256)
        self.conv4 = DepthwiseSeparableConv(256, 512)  # 新增层
        self.fc1 = nn.Linear(512, 256)  # 更新全连接层输入
        self.fc2 = nn.Linear(256, 128)  # 新增全连接层
        self.fc3 = nn.Linear(128, 9)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)
        self.bn6 = nn.BatchNorm1d(128)

    def forward(self, x):
        batchsize = x.size()[0]
#         print("输入第一层", x.shape)
        x = F.elu(self.bn1(self.conv1(x)))
#         print("第一次卷积", x.shape)
        x = F.elu(self.bn2(self.conv2(x)))
#         print("第二次卷积", x.shape)
        x = F.elu(self.bn3(self.conv3(x)))
#         print("第三次卷积", x.shape)
        x = F.elu(self.bn4(self.conv4(x)))  # 新增层
#         print("第四次卷积", x.shape)
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 512)  # 更新全连接层的输入大小
#         print("最大池化", x.shape)
        x = F.elu(self.bn5(self.fc1(x)))
#         print("第一次MLP降维", x.shape)
        x = F.elu(self.bn6(self.fc2(x)))  # 新增全连接层
#         print("第二次MLP降维", x.shape)
        x = self.fc3(x)
#         print("第三次MLP降维", x.shape)
        iden = Variable(torch.eye(3).flatten().to(x.device)).view(1, 9).repeat(batchsize, 1)
        x = x + iden
        x = x.view(-1, 3, 3)
#         print("输出", x.shape)
        return x
class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = DepthwiseSeparableConv(k, 64)
        self.conv2 = DepthwiseSeparableConv(64, 128)
        self.conv3 = DepthwiseSeparableConv(128, 256)  # 增加通道数
        self.conv4 = DepthwiseSeparableConv(256, 512)  # 增加卷积层
        self.conv5 = DepthwiseSeparableConv(512, 1024)  # 增加卷积层
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(1024)
        self.bn6 = nn.BatchNorm1d(512)
        self.bn7 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.elu(self.bn1(self.conv1(x)))
        x = F.elu(self.bn2(self.conv2(x)))
        x = F.elu(self.bn3(self.conv3(x)))
        x = F.elu(self.bn4(self.conv4(x)))  # 新增层
        x = F.elu(self.bn5(self.conv5(x)))  # 新增层
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.elu(self.bn6(self.fc1(x)))
        x = F.elu(self.bn7(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1, self.k * self.k).repeat(batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x

class PointNetfeat(nn.Module):
    def __init__(self, global_feat=False, feature_transform=False):
        super(PointNetfeat, self).__init__()
        self.stn = STN3d()
        self.conv1 = DepthwiseSeparableConv(3, 64)
        self.conv2 = DepthwiseSeparableConv(64, 128)
        self.conv3 = DepthwiseSeparableConv(128, 256)  # 增加通道数
        self.conv4 = DepthwiseSeparableConv(256, 512)  # 增加卷积层
        self.conv5 = DepthwiseSeparableConv(512, 1024)  # 增加卷积层

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        
        if self.feature_transform:
            self.fstn = STNkd(k=64)

    def forward(self, x):
        n_pts = x.size()[2]
        trans = self.stn(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans)
        x = x.transpose(2, 1)
        x = F.elu(self.bn1(self.conv1(x)))  # 使用 ELU
#         print("矩阵相乘后第一次卷积：", x.shape)
        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)
        else:
            trans_feat = None

        pointfeat = x
#         print("保留的特征维度：", x.shape)
        x = F.elu(self.bn2(self.conv2(x)))  # 使用 ELU
#         print("矩阵相乘后第二次卷积：", x.shape)
        x = F.elu(self.bn3(self.conv3(x)))  # 使用 ELU
#         print("矩阵相乘后第三次卷积：", x.shape)
        x = F.elu(self.bn4(self.conv4(x)))  # 使用 ELU
#         print("矩阵相乘后第四次卷积：", x.shape)
        x = F.elu(self.bn5(self.conv5(x)))  # 使用 ELU
#         print("矩阵相乘后第五次卷积：", x.shape)
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
#         print("最大池化：", x.shape)
#         print(self.global_feat)
        if self.global_feat:
            return x, trans, trans_feat

class CTF(nn.Module):
    def __init__(self, k=40, feature_transform=True):
        super(CTF, self).__init__()
        self.feature_transform = feature_transform
        self.feat = PointNetfeat(global_feat=True, feature_transform=feature_transform)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        self.dropout = nn.Dropout(p=0.3)
        
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)

    def forward(self, x):
        x, trans, trans_feat = self.feat(x)
#         print("最后输出", x.shape)
        x = F.elu(self.bn1(self.fc1(x)))  # 使用 ELU
        x = F.elu(self.bn2(self.dropout(self.fc2(x))))  # 使用 ELU
        x = self.fc3(x)
        return F.log_softmax(x, dim=1), trans, trans_feat
    
    def feature_(self, x):
        x, trans, trans_feat = self.feat(x)
        return x

class PointNetDenseCls(nn.Module):
    def __init__(self, k=2, feature_transform=False):
        super(PointNetDenseCls, self).__init__()
        self.k = k
        self.feature_transform = feature_transform
        self.feat = PointNetfeat(global_feat=False, feature_transform=feature_transform)
        self.conv1 = DepthwiseSeparableConv(1088, 512)
        self.conv2 = DepthwiseSeparableConv(512, 256)
        self.conv3 = DepthwiseSeparableConv(256, 128)
        self.conv4 = nn.Conv1d(128, self.k, 1)

        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

    def forward(self, x):
        batchsize = x.size()[0]
        n_pts = x.size()[2]
        x, trans, trans_feat = self.feat(x)
        x = F.elu(self.bn1(self.conv1(x)))  # 使用 ELU
        x = F.elu(self.bn2(self.conv2(x)))  # 使用 ELU
        x = F.elu(self.bn3(self.conv3(x)))  # 使用 ELU
        x = self.conv4(x)
        x = x.transpose(2, 1).contiguous()
        x = F.log_softmax(x.view(-1, self.k), dim=-1)
        x = x.view(batchsize, n_pts, self.k)
        return x, trans, trans_feat

def feature_transform_regularizer(trans):
    d = trans.size()[1]
    batchsize = trans.size()[0]
    I = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        I = I.cuda()
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1)) - I, dim=(1, 2)))
    return loss


# print("?")
# sim_data = torch.rand(32, 3, 2500)
# pointfeat = gf()
# out, _, _ = pointfeat(sim_data)
# print("global feat", out)
