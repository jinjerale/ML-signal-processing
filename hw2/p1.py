import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load the star positions
june = np.load('data/june.npy')
december = np.load('data/december.npy')

# Step 1: Calculate disparity
disparity = december[:, 0] - june[:, 0]  # Only x-coordinates matter
disparity = disparity[:, None]

# Step 2: Plot histogram of the disparity
plt.hist(disparity, bins=30, color='blue', alpha=0.7)
plt.title('Disparity Histogram')
plt.xlabel('Disparity')
plt.ylabel('Frequency')
plt.show()

# Step 3: Perform K-means clustering
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(disparity.reshape(-1, 1))  # Reshape disparity for K-means
labels = kmeans.labels_
cluster_means = kmeans.cluster_centers_

print("K-means Cluster Means: ", cluster_means)

# Analyzing the clusters
if cluster_means[0] < cluster_means[1]:
    galaxy_cluster = 0  # Assume closer stars have larger disparity
    other_galaxy_cluster = 1
else:
    galaxy_cluster = 1
    other_galaxy_cluster = 0

print(f"Cluster {galaxy_cluster} likely corresponds to stars in our galaxy.")
print(f"Cluster {other_galaxy_cluster} likely corresponds to stars in the neighboring galaxy.")

# Step 4: Implement Gaussian Mixture Model (GMM) from scratch

class GMM:
    def __init__(self, n_components, max_iter=100, tol=1e-6):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol

    def initialize_params(self, X):
        np.random.seed(42)
        self.n, self.d = X.shape
        indices = np.random.choice(self.n, self.n_components)
        self.means = X[indices]
        self.weights = np.ones(self.n_components) / self.n_components
        self.variances = np.var(X) * np.ones(self.n_components)

    def gaussian_density(self, X, mean, variance):
        coeff = 1 / np.sqrt(2 * np.pi * variance)
        exponent = -((X - mean) ** 2) / (2 * variance)
        return coeff * np.exp(exponent)

    def e_step(self, X):
        responsibilities = np.zeros((self.n, self.n_components))
        for i in range(self.n_components):
            responsibilities[:, i] = self.weights[i] * self.gaussian_density(X, self.means[i], self.variances[i])
        responsibilities /= responsibilities.sum(axis=1, keepdims=True)
        return responsibilities

    def m_step(self, X, responsibilities):
        Nk = responsibilities.sum(axis=0)
        self.weights = Nk / self.n
        self.means = (responsibilities.T @ X) / Nk
        self.variances = ((responsibilities.T @ (X - self.means[:, np.newaxis])**2) / Nk).flatten()

    def fit(self, X):
        self.initialize_params(X)
        log_likelihood = 0
        for iteration in range(self.max_iter):
            prev_log_likelihood = log_likelihood
            # E-step: Calculate responsibilities
            responsibilities = self.e_step(X)
            # M-step: Update parameters
            self.m_step(X, responsibilities)
            # Check convergence
            log_likelihood = np.sum(np.log(responsibilities.sum(axis=1)))
            if np.abs(log_likelihood - prev_log_likelihood) < self.tol:
                break

    def predict(self, X):
        responsibilities = self.e_step(X)
        return np.argmax(responsibilities, axis=1)

# Step 5: Fit GMM and compare results
gmm = GMM(n_components=2)
gmm.fit(disparity.reshape(-1, 1))
gmm_labels = gmm.predict(disparity.reshape(-1, 1))

# Compare GMM and K-means results
print("GMM Cluster Means: ", gmm.means)
plt.hist(disparity[gmm_labels == 0], bins=25, color='red', alpha=0.5, label='GMM Cluster 0')
plt.hist(disparity[gmm_labels == 1], bins=25, color='green', alpha=0.5, label='GMM Cluster 1')
plt.legend()
plt.title('GMM Cluster Results')
plt.show()

