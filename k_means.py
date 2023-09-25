import torch
import torchvision
from torchvision import transforms
import numpy as np

# Load and preprocess the MNIST dataset using PyTorch
transform = transforms.Compose([transforms.ToTensor()])
mnist_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
X = mnist_dataset.data.view(-1, 28 * 28).float()  # Flatten the images
y_true = mnist_dataset.targets.numpy()

# Number of clusters (adjust as needed)
num_clusters = 40

# Randomly initialize cluster centroids
centroids = X[:num_clusters].clone()

# Maximum number of iterations
max_iterations = 100

for iteration in range(max_iterations):
    # Compute distances between data points and centroids
    distances = torch.norm(X[:, None, :] - centroids[None, :, :], dim=2)
    
    # Assign each data point to the nearest centroid
    cluster_assignments = torch.argmin(distances, dim=1)
    
    # Update centroids by computing the mean of points in each cluster
    new_centroids = torch.stack([X[cluster_assignments == c].mean(dim=0) for c in range(num_clusters)])
    
    # Check for convergence
    if torch.all(torch.eq(centroids, new_centroids)):
        break
    
    centroids = new_centroids

# Map cluster assignments to class labels using a majority voting scheme
cluster_to_class_mapping = {}
for cluster in range(num_clusters):
    cluster_mask = (cluster_assignments == cluster)
    cluster_indices = torch.where(cluster_mask)[0]  # Get indices of data points in the cluster
    if len(cluster_indices) > 0:
        # Find the most common label among data points in the cluster
        most_common_label = torch.bincount(torch.tensor(y_true[cluster_indices])).argmax()
        cluster_to_class_mapping[cluster] = most_common_label.item()

# Map cluster labels to predicted labels
y_pred = np.array([cluster_to_class_mapping[cluster.item()] for cluster in cluster_assignments])

# Evaluate accuracy
accuracy = np.mean(y_true == y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")



mnist_test_dataset = torchvision.datasets.MNIST(
    root='./data', train=False, transform=transform, download=True
)
X_test = mnist_test_dataset.data.view(-1, 28 * 28).float()
y_test_true = mnist_test_dataset.targets.numpy()

# Compute cluster assignments for the test dataset
test_distances = torch.norm(X_test[:, None, :] - centroids[None, :, :], dim=2)
test_cluster_assignments = torch.argmin(test_distances, dim=1)

# Map cluster assignments to class labels using the previously computed mapping
y_test_pred = np.array([cluster_to_class_mapping[cluster.item()] for cluster in test_cluster_assignments])

# Evaluate accuracy on the test dataset
accuracy_test = np.mean(y_test_true == y_test_pred)
print(f"Accuracy on Test Mnist Dataset: {accuracy_test * 100:.2f}%")

import torch
import torchvision
from torchvision import transforms
import numpy as np

# Load and preprocess the CIFAR-10 dataset using PyTorch
transform = transforms.Compose([transforms.ToTensor()])
cifar10_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
X = cifar10_dataset.data  # No need to reshape, data is already 3x32x32
y_true = cifar10_dataset.targets

# Number of clusters (adjust as needed)
num_clusters = 40

# Randomly initialize cluster centroids
centroids = X[:num_clusters].clone().view(num_clusters, -1).float()

# Maximum number of iterations
max_iterations = 100

for iteration in range(max_iterations):
    # Compute distances between data points and centroids
    distances = torch.norm(X[:, None, :, :, :] - centroids[None, :, :, :, :], dim=(2, 3, 4))
    
    # Assign each data point to the nearest centroid
    cluster_assignments = torch.argmin(distances, dim=1)
    
    # Update centroids by computing the mean of points in each cluster
    new_centroids = torch.stack([X[cluster_assignments == c].mean(dim=0) for c in range(num_clusters)])
    
    # Check for convergence
    if torch.all(torch.eq(centroids, new_centroids)):
        break
    
    centroids = new_centroids

# Map cluster assignments to class labels using a majority voting scheme
cluster_to_class_mapping = {}
for cluster in range(num_clusters):
    cluster_mask = (cluster_assignments == cluster)
    cluster_indices = torch.where(cluster_mask)[0]  # Get indices of data points in the cluster
    if len(cluster_indices) > 0:
        # Find the most common label among data points in the cluster
        most_common_label = torch.bincount(torch.tensor(y_true[cluster_indices])).argmax()
        cluster_to_class_mapping[cluster] = most_common_label.item()

# Map cluster labels to predicted labels
y_pred = np.array([cluster_to_class_mapping[cluster.item()] for cluster in cluster_assignments])

# Evaluate accuracy on the training dataset
accuracy = np.mean(y_true == y_pred)
print(f"Accuracy on CIFAR-10 Training Dataset: {accuracy * 100:.2f}%")

# Load and preprocess the CIFAR-10 test dataset
cifar10_test_dataset = torchvision.datasets.CIFAR10(
    root='./data', train=False, transform=transform, download=True
)
X_test = cifar10_test_dataset.data  # No need to reshape, data is already 3x32x32
y_test_true = cifar10_test_dataset.targets

# Compute cluster assignments for the test dataset
test_distances = torch.norm(X_test[:, None, :, :, :] - centroids[None, :, :, :, :], dim=(2, 3, 4))
test_cluster_assignments = torch.argmin(test_distances, dim=1)

# Map cluster assignments to class labels using the previously computed mapping
y_test_pred = np.array([cluster_to_class_mapping[cluster.item()] for cluster in test_cluster_assignments])

# Evaluate accuracy on the test dataset
accuracy_test = np.mean(y_test_true == y_test_pred)
print(f"Accuracy on CIFAR-10 Test Dataset: {accuracy_test * 100:.2f}%")