import logging
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.neighbors import NearestNeighbors
from cpu_knn import Aggregator, NearestNeighbourScorer, FaissNN, IdentitySampler
from feature import Feature
from tqdm import tqdm
from sklearn.cluster import KMeans
LOGGER = logging.getLogger(__name__)

def fill_missing_values(x_data, x_label, y_data, k=1):
    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(x_data)
    distances, indices = nn.kneighbors(y_data)
    avg_values = np.mean(x_label[indices], axis=1)
    return avg_values

class PatchCore(torch.nn.Module):
    def __init__(self, device, num_groups=128,feature_extractor_type="dgcnn"):
        super(PatchCore, self).__init__()
        self.device = device
        self.feature_extractor_type = feature_extractor_type
        self.num_groups = num_groups
        self.deep_feature_extractor = None
        self.set_feature_extractor()
        self.aggregator = Aggregator(target_dim=256).to(self.device)
        self.feature_sampler = IdentitySampler()
        self.anomaly_scorer = NearestNeighbourScorer(
            n_nearest_neighbours=1,
            nn_method=FaissNN(False, 4)
        )
        self.memory_features = None
        self.memory_centers = None
    def set_feature_extractor(self):
        if self.feature_extractor_type == "dgcnn":
            self.deep_feature_extractor = Feature(model_type="dgcnn", num_groups=self.num_groups, group_size=512)
        elif self.feature_extractor_type == "pointnet":
            self.deep_feature_extractor = Feature(model_type="pointnet", num_groups=self.num_groups, group_size=512)
        elif self.feature_extractor_type == "pointnet2":
            self.deep_feature_extractor = Feature(model_type="pointnet2", num_groups=self.num_groups, group_size=512)
        elif self.feature_extractor_type == "pointmlp":
            self.deep_feature_extractor = Feature(model_type="pointmlp", num_groups=self.num_groups, group_size=512)
        elif self.feature_extractor_type == "CTF":
            self.deep_feature_extractor = Feature(model_type="CTF", num_groups=self.num_groups, group_size=128)
        else:
            raise ValueError(f"Unsupported feature extractor type: {self.feature_extractor_type}")
        self.deep_feature_extractor = self.deep_feature_extractor.to(self.device)

    def embed(self, data_loader):
        features = []
        coordinates = []
        for batch in tqdm(data_loader, desc="Processing Batches"):
            images = batch[0].to(torch.float).to(self.device).squeeze(1)

            with torch.no_grad():
                batch_features, center_coords = self._extract_features(images)
                features.append(batch_features)
                coordinates.append(center_coords)
        return np.concatenate(features, axis=0), np.concatenate(coordinates, axis=0)

    def _extract_features(self, input_data):
        features, center_coords = self.deep_feature_extractor(input_data)
        features = features.cpu().numpy().astype(np.float32)
        center_coords = center_coords.cpu().numpy()
        return features, center_coords

    def fit(self, training_data_loader):
        features, centers = self.embed(training_data_loader)
        self.memory_features = self.feature_sampler.run(features)
        self.memory_centers = centers
        self.anomaly_scorer.fit(detection_features=[self.memory_features])

    def predict(self, data_loader):
        sample_scores = []
        all_point_scores = []
        labels_gt = []
        masks_gt = []

        for batch in tqdm(data_loader, desc="Processing Batches"):
            images = batch[0].to(torch.float).to(self.device)
            labels = batch[2]  
            masks = batch[1]   

            with torch.no_grad():
                features, centers = self._extract_features(images.squeeze(1))
                center_scores = self.anomaly_scorer.predict([features])[0]
                images = images.squeeze(0)

                full_scores = fill_missing_values(
                    x_data=centers,          
                    x_label=center_scores,  
                    y_data=images.squeeze(0).cpu().numpy().T,  
                    k=10
                )
                full_scores_tensor = torch.tensor(full_scores)
                full_scores = full_scores_tensor.reshape(1, -1)
                full_scores = full_scores.numpy()
                sample_scores.append(np.max(full_scores))
                all_point_scores.append(full_scores)
                labels_gt.append(labels.cpu().numpy())
                masks_gt.append(masks.cpu().numpy())
        return sample_scores, all_point_scores, labels_gt, masks_gt
