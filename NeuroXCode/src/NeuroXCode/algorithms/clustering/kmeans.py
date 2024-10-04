from sklearn.cluster import KMeans
from ...utilities.utils import load_data, save_clustering_results, log_clustering_process
import time

class KMeansClusteringPipeline:
    def __init__(self, num_clusters=5):
        self.num_clusters = num_clusters

    @staticmethod
    def load_and_prepare_data(point_file=None, vocab_file=None, num_points=100, num_dims=5, vocab_size=100):
        """Load or generate synthetic data."""
        points, vocab = load_data(point_file, vocab_file, num_points, num_dims, vocab_size)
        return points, vocab

    def perform_kmeans_clustering(self, data):
        """Perform K-Means clustering on the input data."""
        kmeans = KMeans(n_clusters=self.num_clusters, verbose=3)
        kmeans.fit(data)
        return kmeans

    def run_pipeline(self, points, vocab):
        """Run the full K-Means clustering pipeline."""
        start_time = time.time()

        # Perform K-Means clustering
        clustering = self.perform_kmeans_clustering(points)

        # Create a dictionary of clusters with words from vocab
        clusters = {i: [vocab[idx] for idx in range(len(vocab)) if clustering.labels_[idx] == i]
                    for i in range(self.num_clusters)}

        end_time = time.time()

        # Return the clustering and the cluster assignments
        return clustering, clusters, end_time - start_time
