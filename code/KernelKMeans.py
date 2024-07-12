import numpy as np
from sklearn.metrics.pairwise import pairwise_kernels

class KernelKMeans:
    def __init__(self, n_clusters=3, kernel="rbf", max_iter=100, tol=1e-6, gamma=None):
        self.n_clusters = n_clusters
        self.kernel = kernel
        self.max_iter = max_iter
        self.tol = tol
        self.gamma = gamma
    
    def fit(self, X):
        self.X_fit_ = X
        n_samples = X.shape[0]
        self.labels_ = np.random.randint(self.n_clusters, size=n_samples)
        self.K = pairwise_kernels(X, metric=self.kernel, gamma=self.gamma)
        self.distance = np.zeros((n_samples, self.n_clusters))

        for _ in range(self.max_iter):
            self._compute_distances()
            new_labels = np.argmin(self.distance, axis=1)

            if np.sum(self.labels_ != new_labels) < self.tol * n_samples:
                break

            self.labels_ = new_labels
        
        return self

    def _compute_distances(self):
        for k in range(self.n_clusters):
            mask = self.labels_ == k
            n_k = np.sum(mask)
            if n_k == 0:
                continue
            K_k = self.K[mask][:, mask]
            distance_to_cluster = np.sum(self.K[:, mask], axis=1) / n_k
            distance_within_cluster = np.sum(K_k) / (n_k ** 2)
            self.distance[:, k] = np.diag(self.K) - 2 * distance_to_cluster + distance_within_cluster
    
    def predict(self, X):
        K = pairwise_kernels(X, self.X_fit_, metric=self.kernel, gamma=self.gamma)
        distances = np.zeros((X.shape[0], self.n_clusters))
        for k in range(self.n_clusters):
            mask = self.labels_ == k
            n_k = np.sum(mask)
            if n_k == 0:
                continue
            distance_to_cluster = np.sum(K[:, mask], axis=1) / n_k
            distance_within_cluster = np.sum(self.K[mask][:, mask]) / (n_k ** 2)
            distances[:, k] = np.diag(K) - 2 * distance_to_cluster + distance_within_cluster
        return np.argmin(distances, axis=1)

# Ejemplo de uso:
if __name__ == "__main__":
    from sklearn.datasets import make_moons
    import matplotlib.pyplot as plt
    import matplotlib

    matplotlib.use("TkAgg")

    X, _ = make_moons(n_samples=1000, noise=0.1)

    model = KernelKMeans(n_clusters=2, kernel="rbf", gamma=1.0)
    model.fit(X)
    labels = model.labels_

    plt.scatter(X[:, 0], X[:, 1], c=labels)
    plt.title("Kernel K-Means Clustering")
    plt.show()
