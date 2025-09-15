import os
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from data import ModelNet40
from model.dgcnn import DGCNN, PointNet
from model.pointnet import PointNetCls
from util import IOStream
from model.pointmlp import pointMLP

def test(args, io):
    # 数据加载
    test_loader = DataLoader(ModelNet40(partition='test', num_points=args.num_points, use_normals=False),
                             batch_size=args.test_batch_size, shuffle=False, drop_last=False)

    device = torch.device("cuda" if args.cuda else "cpu")

    # 加载模型并加载预训练权重
    model = pointMLP().to(device) if args.model == 'pointMLP' else PointNet(args).to(device)
    model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load(f'pretrained/pointmlp.t7'))
    model.eval()

    test_true, test_pred = [], []

    # 测试阶段
    with torch.no_grad():
        for data, label in test_loader:
            data, label = data.to(device), label.to(device).squeeze()
            data = data.permute(0, 2, 1)
            logits = model(data)
            preds = logits.max(dim=1)[1]

            test_true.append(label.cpu().numpy())
            test_pred.append(preds.detach().cpu().numpy())

    # 计算测试集上的准确率和平均每类准确率
    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)
    test_acc = accuracy_score(test_true, test_pred)
    avg_per_class_acc = balanced_accuracy_score(test_true, test_pred)

    # 输出测试结果
    io.cprint(f'Test accuracy: {test_acc:.6f}, Average per-class accuracy: {avg_per_class_acc:.6f}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--model', type=str, default='pointMLP', metavar='N',
                        choices=['pointnet', 'dgcnn'],
                        help='Model to use, [pointnet, dgcnn]')
    parser.add_argument('--num_points', type=int, default=512,
                        help='num of points to use')
    parser.add_argument('--test_batch_size', type=int, default=16, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')
    parser.add_argument('--k', type=int, default=20, metavar='N',
                        help='Num of nearest neighbors to use')    
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')
    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    io = IOStream(f'checkpoints/{args.exp_name}/test.log')
    io.cprint(str(args))

    # 运行测试
    test(args, io)
