import torch
def process_single_point_cloud(point_cloud):
    if not isinstance(point_cloud, torch.Tensor):
        point_cloud = torch.tensor(point_cloud)

    N, _ = point_cloud.shape

    centroid = point_cloud.mean(dim=0, keepdim=True)  # [1, 3]
    centered_cloud = point_cloud - centroid  # [N, 3]

    distances = torch.norm(centered_cloud, dim=1)  # [N]

    max_dist_idx = torch.argmax(distances)  # []
    min_dist_idxs = torch.topk(distances, k=2, largest=False).indices  # [2]

    farthest_point = point_cloud[max_dist_idx]  # [3]
    closest_point1 = point_cloud[min_dist_idxs[0]]  # [3]
    closest_point2 = point_cloud[min_dist_idxs[1]]  # [3]

    v1 = (farthest_point - centroid.squeeze()).unsqueeze(-1)  # [3, 1]
    v1 = v1 / torch.norm(v1)
    
    v2 = (closest_point1 - centroid.squeeze()).unsqueeze(-1)
    v2 = v2 - (v2 * v1).sum() * v1
    v2 = v2 / torch.norm(v2)

    v3 = (closest_point2 - centroid.squeeze()).unsqueeze(-1)
    v3 = v3 - (v3 * v1).sum() * v1 - (v3 * v2).sum() * v2
    v3 = v3 / torch.norm(v3)

    basis = torch.cat([v1, v2, v3], dim=1)  # [3, 3]

    transformed_data = torch.matmul(basis.T, centered_cloud.T).T  # [N, 3]
    transformed_data[torch.isnan(transformed_data)] = 0
    return transformed_data

def process_point_clouds(batch_cloud):
    B, _, N = batch_cloud.shape
    
    centroids = batch_cloud.mean(dim=2, keepdim=True)  # [B, 3, 1]
    centered_cloud = batch_cloud - centroids  # [B, 3, N]

    distances = torch.norm(centered_cloud, dim=1)  # [B, N]

    max_dist_idxs = torch.argmax(distances, dim=1)  # [B]
    min_dist_idxs = torch.topk(distances, k=2, largest=False).indices  # [B, 2]

    farthest_points = batch_cloud[torch.arange(B), :, max_dist_idxs]  # [B, 3]
    closest_points1 = batch_cloud[torch.arange(B), :, min_dist_idxs[:, 0]]  # [B, 3]
    closest_points2 = batch_cloud[torch.arange(B), :, min_dist_idxs[:, 1]]  # [B, 3]

    v1 = (farthest_points - centroids.squeeze()).unsqueeze(-1)  # [B, 3, 1]
    v1 = v1 / torch.norm(v1, dim=1, keepdim=True)
    
    v2 = (closest_points1 - centroids.squeeze()).unsqueeze(-1)
    v2 = v2 - (v2 * v1).sum(dim=1, keepdim=True) * v1
    v2 = v2 / torch.norm(v2, dim=1, keepdim=True)

    v3 = (closest_points2 - centroids.squeeze()).unsqueeze(-1)
    v3 = v3 - (v3 * v1).sum(dim=1, keepdim=True) * v1 - (v3 * v2).sum(dim=1, keepdim=True) * v2
    v3 = v3 / torch.norm(v3, dim=1, keepdim=True)

    basis = torch.cat([v1, v2, v3], dim=2)  # [B, 3, 3]

    transformed_data = torch.matmul(basis.transpose(1, 2), centered_cloud)
    return transformed_data


