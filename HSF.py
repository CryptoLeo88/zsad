from typing import Union, List, Optional
import numpy as np
import torch
from pkg_resources import packaging
from torch import nn
from torch.nn import functional as F
from sklearn.cluster import KMeans
class HybridSemanticFusion(nn.Module):
    def __init__(self, k_clusters = 20):
        super(HybridSemanticFusion, self).__init__()
        self.k_clusters = k_clusters
        self.n_aggregate_patch_tokens = k_clusters * 5
        self.cluster_performer = KMeans(n_clusters=self.k_clusters, n_init="auto")

    # @torch.no_grad()
    def forward(self, patch_tokens: list, anomaly_maps: list):
        anomaly_map = torch.mean(torch.stack(anomaly_maps, dim=1), dim=1)
        anomaly_map = torch.softmax(anomaly_map, dim=2)[:, :, 1] # B, L

        # extract most abnormal feats
        selected_abnormal_tokens = []
        k = min(anomaly_map.shape[1], self.n_aggregate_patch_tokens)
        top_k_indices = torch.topk(anomaly_map, k=k, dim=1).indices
        for layer in range(len(patch_tokens)):
            selected_tokens = patch_tokens[layer]. \
                gather(dim=1, index=top_k_indices.unsqueeze(-1).
                       expand(-1, -1, patch_tokens[layer].shape[-1]))
            selected_abnormal_tokens.append(selected_tokens)

        # use kmeans to extract these centriods
        # Stack the data_preprocess
        stacked_data = torch.cat(selected_abnormal_tokens, dim=2)

        batch_cluster_centers = []
        # Perform K-Means clustering
        for b in range(stacked_data.shape[0]):
            cluster_labels = self.cluster_performer.fit_predict(stacked_data[b, :, :].detach().cpu().numpy())

            # Initialize a list to store the cluster centers
            cluster_centers = []

            # Extract cluster centers for each cluster
            for cluster_id in range(self.k_clusters):
                collected_cluster_data = []
                for abnormal_tokens in selected_abnormal_tokens:
                    cluster_data = abnormal_tokens[b, :, :][cluster_labels == cluster_id]
                    collected_cluster_data.append(cluster_data)
                collected_cluster_data = torch.cat(collected_cluster_data, dim=0)
                cluster_center = torch.mean(collected_cluster_data, dim=0, keepdim=True)
                cluster_centers.append(cluster_center)

            # Normalize the cluster centers
            cluster_centers = torch.cat(cluster_centers, dim=0)
            cluster_centers = torch.mean(cluster_centers, dim=0)
            batch_cluster_centers.append(cluster_centers)

        batch_cluster_centers = torch.stack(batch_cluster_centers, dim=0)
        batch_cluster_centers = F.normalize(batch_cluster_centers, dim=1)

        return batch_cluster_centers

        # # preprocess
        # # compute the anomaly map
        # anomaly_map = torch.mean(torch.stack(anomaly_maps, dim=1), dim=1)
        # anomaly_map = torch.softmax(anomaly_map, dim=2)[:, :, 1] # B, L
        #
        # # compute the average multi-hierarchy patch embeddings
        # avg_patch_tokens = torch.mean(torch.stack(patch_tokens, dim=0), dim=0) # B, L, C
        #
        # # Initialize a list to store the centroids of clusters with the largest anomaly scores
        # cluster_centroids = []
        #
        # # loop across the batch size
        # for b in range(avg_patch_tokens.shape[0]):
        #     # step1: group features into clusters
        #     cluster_labels = self.cluster_performer.fit_predict(avg_patch_tokens[b, :, :].detach().cpu().numpy())
        #
        #     # step2: compute the anomaly scores for individual clusters via the anomaly map
        #     # Convert cluster labels back to tensor
        #     cluster_labels = torch.tensor(cluster_labels).to(avg_patch_tokens.device)
        #     cluster_anomaly_scores = {}
        #     for label in torch.unique(cluster_labels):
        #         cluster_indices = torch.where(cluster_labels == label)[0]
        #         cluster_anomaly_scores[label.item()] = anomaly_map[b, cluster_indices].mean().item()
        #
        #     # step3: select the cluster with the largest anomaly score and then compute its centroid by averaging the
        #     # corresponding avg_patch_tokens
        #     # Find the cluster with the largest anomaly score
        #     largest_anomaly_cluster = max(cluster_anomaly_scores, key=cluster_anomaly_scores.get)
        #
        #     # Get the indices of the tokens belonging to the largest anomaly cluster
        #     largest_anomaly_cluster_indices = torch.where(cluster_labels == largest_anomaly_cluster)[0]
        #
        #     # Compute the centroid of the largest anomaly cluster by averaging the corresponding avg_patch_tokens
        #     centroid = avg_patch_tokens[b, largest_anomaly_cluster_indices, :].mean(dim=0)
        #
        #     # Append the centroid to the list of cluster centroids
        #     cluster_centroids.append(centroid)
        #
        # # Convert the list of centroids to a tensor
        # cluster_centroids = torch.stack(cluster_centroids, dim=0)
        # cluster_centroids = F.normalize(cluster_centroids, dim=1)

        # return cluster_centroids