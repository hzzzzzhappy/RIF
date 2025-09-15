#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: main.py
@Time: 2018/10/13 10:39 PM
"""


from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from data import ModelNet40
from model.dgcnn import PointNet, DGCNN
from model.pointnet import PointNetCls
import numpy as np
from torch.utils.data import DataLoader
from util import cal_loss, IOStream
import sklearn.metrics as metrics
import matplotlib.pyplot as plt

def plot_loss_curve(train_losses, test_losses, epoch, save_path='checkpoints/loss_curve.png'):
    """
    绘制并保存损失曲线图。
    
    参数:
    - train_losses: list, 每个记录点的训练损失。
    - test_losses: list, 每个记录点的测试损失。
    - epoch: 当前 epoch，用于在图中显示。
    - save_path: str, 图像保存路径。
    """
    plt.figure()
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="Train Loss")
    plt.plot(range(1, len(test_losses) + 1), test_losses, label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Loss Curve - Epoch {epoch}")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
    
def load_pretrained_model(model, pretrained_path, device):
    """
    加载预训练模型，并去掉头层（通常是最后的全连接层）。
    
    参数:
    - model: nn.Module, 当前模型对象。
    - pretrained_path: str, 预训练模型的路径。
    - device: torch.device, 指定模型加载的设备（CPU 或 GPU）。
    
    返回:
    - model: nn.Module, 加载了预训练权重并去掉头层的模型。
    """
    # 加载预训练的权重
    checkpoint = torch.load(pretrained_path, map_location=device)
    
    # 去掉头层（假设头层的名称是 'fc' 或类似）
    # 通过遍历 state_dict，可以检查并去掉需要去掉的层
    modified_state_dict = {k: v for k, v in checkpoint.items() if not k.startswith('fc')}
    
    # 将修改后的权重加载到模型
    model.load_state_dict(modified_state_dict, strict=False)
    print("Loaded pretrained model with head removed.")
    return model


def _init_():
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/'+args.exp_name):
        os.makedirs('checkpoints/'+args.exp_name)
    if not os.path.exists('checkpoints/'+args.exp_name+'/'+'models'):
        os.makedirs('checkpoints/'+args.exp_name+'/'+'models')
    os.system('cp main.py checkpoints'+'/'+args.exp_name+'/'+'main.py.backup')
    os.system('cp model.py checkpoints' + '/' + args.exp_name + '/' + 'model.py.backup')
    os.system('cp util.py checkpoints' + '/' + args.exp_name + '/' + 'util.py.backup')
    os.system('cp data.py checkpoints' + '/' + args.exp_name + '/' + 'data.py.backup')

def train(args, io):
    train_loader = DataLoader(ModelNet40(partition='train', num_points=args.num_points), num_workers=8,
                              batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(ModelNet40(partition='test', num_points=args.num_points), num_workers=8,
                             batch_size=args.test_batch_size, shuffle=True, drop_last=False)

    device = torch.device("cuda" if args.cuda else "cpu")

    # Initialize the model
    if args.model == 'pointnet':
        model = PointNet(args).to(device)
    elif args.model == 'pointnetcls':
        model = PointNetCls().to(device)
    else:
        raise Exception("Not implemented")
    
    model = nn.DataParallel(model)

    # Initialize optimizer and scheduler
    if args.use_sgd:
        opt = optim.SGD(model.parameters(), lr=args.lr * 100, momentum=args.momentum, weight_decay=1e-4)
    else:
        opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=args.lr)
    criterion = cal_loss

    best_test_acc = 0
    train_losses = []
    test_losses = []

    for epoch in range(args.epochs):
        scheduler.step()

        ####################
        # Train
        ####################
        train_loss = 0.0
        count = 0.0
        model.train()
        train_pred = []
        train_true = []
        
        for batch_idx, (data, label) in enumerate(train_loader):
            data, label = data.to(device), label.to(device).squeeze()
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            opt.zero_grad()
            
            # 检查数据中的 NaN 或无穷值
            if torch.isnan(data).any() or torch.isinf(data).any():
                print(f"NaN or Inf found in input data at batch {batch_idx}")
                continue
            
            logits, _, _ = model(data)
            
            # 检查预测输出中的 NaN 或无穷值
            if torch.isnan(logits).any() or torch.isinf(logits).any():
                print(f"NaN or Inf found in model output at batch {batch_idx}")
                continue

            loss = criterion(logits, label)
            
            # 检查损失值是否为 NaN 或无穷
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"NaN or Inf found in loss at batch {batch_idx}")
                continue

            loss.backward()
            
            # 检查梯度中的 NaN 或无穷值
            for name, param in model.named_parameters():
                if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                    print(f"NaN or Inf found in gradients for {name} at batch {batch_idx}")
                    continue

            opt.step()

            preds = logits.max(dim=1)[1]
            count += batch_size
            train_loss += loss.item() * batch_size
            train_true.append(label.cpu().numpy())
            train_pred.append(preds.detach().cpu().numpy())

        train_true = np.concatenate(train_true)
        train_pred = np.concatenate(train_pred)
        train_loss_avg = train_loss / count
        train_losses.append(train_loss_avg)
        train_acc = metrics.accuracy_score(train_true, train_pred)
        train_avg_acc = metrics.balanced_accuracy_score(train_true, train_pred)

        outstr = 'Train %d, loss: %.6f, train acc: %.6f, train avg acc: %.6f' % (epoch, train_loss_avg, train_acc, train_avg_acc)
        io.cprint(outstr)

        ####################
        # Test
        ####################
        test_loss = 0.0
        count = 0.0
        model.eval()
        test_pred = []
        test_true = []
        
        for data, label in test_loader:
            data, label = data.to(device), label.to(device).squeeze()
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            
            with torch.no_grad():
                logits, _, _ = model(data)
                loss = criterion(logits, label)
            
            preds = logits.max(dim=1)[1]
            count += batch_size
            test_loss += loss.item() * batch_size
            test_true.append(label.cpu().numpy())
            test_pred.append(preds.detach().cpu().numpy())

        test_true = np.concatenate(test_true)
        test_pred = np.concatenate(test_pred)
        test_loss_avg = test_loss / count
        test_losses.append(test_loss_avg)
        test_acc = metrics.accuracy_score(test_true, test_pred)
        test_avg_acc = metrics.balanced_accuracy_score(test_true, test_pred)

        outstr = 'Test %d, loss: %.6f, test acc: %.6f, test avg acc: %.6f' % (epoch, test_loss_avg, test_acc, test_avg_acc)
        io.cprint(outstr)

        # 每 10 个 epoch 更新并保存一次损失图
        if (epoch + 1) % 10 == 0:
            plot_loss_curve(train_losses, test_losses, epoch + 1, save_path='loss_curve.svg')

        # 保存最佳模型
        if test_acc >= best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), 'checkpoints/%s/models/pointnet.t7' % args.exp_name)
def test(args, io):
    test_loader = DataLoader(ModelNet40(partition='test', num_points=args.num_points),
                             batch_size=args.test_batch_size, shuffle=True, drop_last=False)

    device = torch.device("cuda" if args.cuda else "cpu")

    #Try to load models
    model = DGCNN(args).to(device)
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(args.model_path))
    model = model.eval()
    test_acc = 0.0
    count = 0.0
    test_true = []
    test_pred = []
    for data, label in test_loader:

        data, label = data.to(device), label.to(device).squeeze()
        data = data.permute(0, 2, 1)
        batch_size = data.size()[0]
        logits = model(data)
        preds = logits.max(dim=1)[1]
        test_true.append(label.cpu().numpy())
        test_pred.append(preds.detach().cpu().numpy())
    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)
    test_acc = metrics.accuracy_score(test_true, test_pred)
    avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
    outstr = 'Test :: test acc: %.6f, test avg acc: %.6f'%(test_acc, avg_per_class_acc)
    io.cprint(outstr)


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--model', type=str, default='pointnetcls', metavar='N',
                        choices=['pointnet', 'pointnetcls'],
                        help='Model to use, [pointnet, dgcnn]')
    parser.add_argument('--dataset', type=str, default='modelnet40', metavar='N',
                        choices=['modelnet40'])
    parser.add_argument('--batch_size', type=int, default=32, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=16, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=200, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--use_sgd', type=bool, default=True,
                        help='Use SGD')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--eval', type=bool,  default=False,
                        help='evaluate the model')
    parser.add_argument('--num_points', type=int, default=512,
                        help='num of points to use')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--k', type=int, default=20, metavar='N',
                        help='Num of nearest neighbors to use')
    parser.add_argument('--model_path', type=str, default='', metavar='N',
                        help='Pretrained model path')
    args = parser.parse_args()

    _init_()

    io = IOStream('checkpoints/' + args.exp_name + '/run.log')
    io.cprint(str(args))

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        io.cprint(
            'Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')
        torch.cuda.manual_seed(args.seed)
    else:
        io.cprint('Using CPU')

    if not args.eval:
        train(args, io)
    else:
        test(args, io)