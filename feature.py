import torch
import torch.nn as nn
from sklearn.neighbors import NearestNeighbors
from model.dgcnn import DGCNN  
from model.pointnet import PointNetCls
from model.pointnet2 import pointnet2_cls_msg
from model.pointmlp import pointMLP  
from model.CTF import CTF
from util_.trans import PCM

class Feature(nn.Module):
    def __init__(self, model_type="gf", num_groups=1, group_size=512):
        super(Feature, self).__init__()
        self.num_groups = num_groups
        self.group_size = group_size
        self.model_type = model_type

        if model_type == "dgcnn":
            self.feature_extractor = DGCNN()
            self.feature_dim = 2048
        elif model_type == "pointnet":
            self.feature_extractor = PointNetCls()
            self.feature_dim = 1024 
        elif model_type == "pointnet2":
            self.feature_extractor = pointnet2_cls_msg()
            self.feature_dim = 1024 
        elif model_type == "pointmlp":
            self.feature_extractor = pointMLP()
            self.feature_dim = 1024
        elif model_type == "gf":
            self.feature_extractor = CTF()
            self.feature_dim = 1024
        else:
            raise ValueError(f"unexpected type: {model_type}")

        self._load_pretrained_weights()

    def _load_pretrained_weights(self):
        weight_path = f'pretrained/{self.model_type}.t7'
        try:
            state_dict = torch.load(weight_path)
            if any(key.startswith("module.") for key in state_dict.keys()):
                state_dict = {key.replace("module.", ""): value for key, value in state_dict.items()}

            self.feature_extractor.load_state_dict(state_dict)
        except Exception as e:
            print(f"error: {e}")

    def _knn_points(self, centroids, points, K):
        nbrs = NearestNeighbors(n_neighbors=K, algorithm='auto').fit(points.cpu().numpy())
        distances, indices = nbrs.kneighbors(centroids.cpu().numpy())
        indices = torch.from_numpy(indices).long()
        return indices

    def forward(self, point_cloud):
        B, _, N = point_cloud.shape  
        point_cloud = point_cloud.squeeze(0).permute(1, 0)  # [3, N] -> [N, 3]

        centroid = point_cloud.mean(dim=0, keepdim=True)  # [1, 3]

        distances = torch.norm(point_cloud - centroid, dim=1)  # [N]
        farthest_point = point_cloud[distances.argmax()].unsqueeze(0)  # [1, 3]

        sampled_points = [farthest_point]
        for _ in range(self.num_groups - 1):
            dist_to_sampled = torch.stack([torch.norm(point_cloud - p, dim=1) for p in sampled_points], dim=0)
            min_dist_to_sampled = dist_to_sampled.min(dim=0).values
            next_point = point_cloud[min_dist_to_sampled.argmax()].unsqueeze(0)
            sampled_points.append(next_point)

        sampled_points = torch.cat(sampled_points, dim=0)  # [num_groups, 3]

        knn_indices = self._knn_points(sampled_points, point_cloud, self.group_size)  # [num_groups, group_size]

        patch_features = []
        i = 0
        while i < self.num_groups - 1:
            patch_points_1 = PCM(point_cloud[knn_indices[i]]).permute(1, 0).unsqueeze(0)  # [1, 3, group_size]
            patch_points_2 = PCM(point_cloud[knn_indices[i + 1]]).permute(1, 0).unsqueeze(0)  # [1, 3, group_size]
            patch_points_1 = point_cloud[knn_indices[i]].permute(1, 0).unsqueeze(0)  # [1, 3, group_size]
            patch_points_2 = point_cloud[knn_indices[i + 1]].permute(1, 0).unsqueeze(0)  # [1, 3, group_size]

            batch_patch_points = torch.cat([patch_points_1, patch_points_2], dim=0)  # [2, 3, group_size]
            batch_features = self.feature_extractor.feature_(batch_patch_points)  # [2, feature_dim]
            patch_features.append(batch_features)
            i += 2
        if self.num_groups % 2 == 1:
            patch_points_last = PCM(point_cloud[knn_indices[-1]]).permute(1, 0).unsqueeze(0)  # [1, 3, group_size]
            batch_patch_points_last = torch.cat([patch_points_last, patch_points_last], dim=0)  # [2, 3, group_size]

            last_feature = self.feature_extractor.feature_(batch_patch_points_last)[0:1]  # [1, feature_dim]
            patch_features.append(last_feature)
        patch_features = torch.cat(patch_features, dim=0).unsqueeze(0)  # [1, num_groups, feature_dim]
        sampled_points = sampled_points.unsqueeze(0)  # [1, num_groups, 3]

        return patch_features.squeeze(0), sampled_points.squeeze(0) 

