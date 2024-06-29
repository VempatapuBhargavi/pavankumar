from sklearn.metrics import silhouette_score, davies_bouldin_score
import torch
import matplotlib.pyplot as plt
from sklearn.utils.multiclass import unique_labels


# Calculate Inertia

def calculate_inertia(X, labels):
 unique_labels = torch.unique(labels)
 inertia = 0.0
for label in unique_labels:
 if label == -1: # Skip noise points
 continue
 cluster_points = X[labels == label]
 centroid = cluster_points.mean(dim=0)
 inertia += torch.sum((cluster_points - centroid) ** 2).item()
 return inertia
# Calculate Silhouette Score
def calculate_silhouette_score(X, labels):
 return silhouette_score(X.numpy(), labels.numpy())
# Calculate Davies-Bouldin Index
def calculate_davies_bouldin_index(X, labels):
 return davies_bouldin_score(X.numpy(), labels.numpy())
# Calculate evaluation metrics
inertia = calculate_inertia(X_tensor, labels)
silhouette = calculate_silhouette_score(X_tensor, labels)
davies_bouldin = calculate_davies_bouldin_index(X_tensor, labels)
print(f'Inertia: {inertia:.2f}')
print(f'Silhouette Score: {silhouette:.2f}')
print(f'Davies-Bouldin Index: {davies_bouldin:.2f}')
#Visualizing Clustering Results
# Plot the clusters
def plot_clusters(X, labels):
 unique_labels = torch.unique(labels)
 colors = plt.get_cmap("viridis", len(unique_labels))
 plt.figure(figsize=(8, 6))
for k in unique_labels:
  if k == -1:
   color = 'k'
   marker = 'x'
 edgecolor = 'none'
 else:
 color = colors(k / len(unique_labels))
 marker = 'o'
 edgecolor = 'k'
 class_member_mask = (labels == k)
 xy = X[class_member_mask]
 plt.scatter(xy[:, 0], xy[:, 1], c=[color], marker=marker, edgecolor=edgecolor, s=50,
label=f'Cluster {k}' if k != -1 else 'Noise')
 plt.title('DBSCAN Clustering')
 plt.xlabel('Feature 1')
 plt.ylabel('Feature 2')
 plt.legend()
 plt.show()
plot_clusters(X_tensor.numpy(), labels)