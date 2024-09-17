from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
from conceptx.utilities.utils import load_data

class AgglomerativeClusteringPipeline:
    def __init__(self, num_clusters=5):
        self.num_clusters = num_clusters

    @staticmethod
    def load_and_prepare_data(point_file=None, vocab_file=None, num_points=100, num_dims=5, vocab_size=100):
        """Use the functional approach to load or generate synthetic data."""
        points, vocab = load_data(point_file, vocab_file, num_points, num_dims, vocab_size)
        return points, vocab

    def perform_agglomerative_clustering(self, data):
        """Perform agglomerative clustering on the input data."""
        clustering = AgglomerativeClustering(n_clusters=self.num_clusters)
        clustering.fit(data)
        return clustering

    @staticmethod
    def create_linkage_matrix(data):
        """Create a linkage matrix using Ward's method."""
        linkage_matrix = linkage(data, method='ward')
        return linkage_matrix

    @staticmethod
    def plot_dendrogram(linkage_matrix):
        """Plot the dendrogram for the linkage matrix and return the figure."""
        plt.figure(figsize=(10, 7))
        dendrogram(linkage_matrix)
        plt.title('Agglomerative Clustering Dendrogram')
        plt.xlabel('Sample index')
        plt.ylabel('Distance')
        plt.show()

    def run_pipeline(self, points, vocab):
        """Run the full clustering pipeline."""
        clustering = self.perform_agglomerative_clustering(points)
        clusters = {i: [vocab[idx] for idx in range(len(vocab)) if clustering.labels_[idx] == i] for i in range(self.num_clusters)}
        return clustering, clusters