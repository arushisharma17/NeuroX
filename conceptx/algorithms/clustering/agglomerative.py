import os
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import matplotlib.pyplot as plt
from conceptx.utilities.utils import load_data, save_clustering_results, generate_synthetic_data

class AgglomerativeClusteringPipeline:
    def __init__(self, output_path='./output', num_clusters=5):
        self.output_path = output_path
        self.num_clusters = num_clusters
        os.makedirs(self.output_path, exist_ok=True)
    def load_and_prepare_data(self, point_file=None, vocab_file=None, num_points=100, num_dims=5, vocab_size=100):
        """Use the functional approach to load or generate synthetic data."""
        points, vocab = load_data(point_file, vocab_file, num_points, num_dims, vocab_size, self.output_path)
        return points, vocab
    def perform_agglomerative_clustering(self, data):
        """Perform agglomerative clustering on the input data using SciPy."""
        linkage_matrix = self.create_linkage_matrix(data)
        labels = fcluster(linkage_matrix, t=self.num_clusters, criterion='maxclust') - 1
        return labels, linkage_matrix
    def create_linkage_matrix(self, data):
        """Create a linkage matrix using Ward's method."""
        linkage_matrix = linkage(data, method='ward')
        return linkage_matrix
    def plot_dendrogram(self, linkage_matrix, file_name):
        """Plot the dendrogram for the linkage matrix."""
        plt.figure(figsize=(10, 7))
        dendrogram(linkage_matrix)
        plt.title('Agglomerative Clustering Dendrogram')
        plt.xlabel('Sample index')
        plt.ylabel('Distance')
        plt.savefig(f"{self.output_path}/{file_name}")
        plt.close()
    def save_clustering(self, clustering, clusters, ref=''):
        """Save the clustering results using the save_clustering_results function from utils.py."""
        save_clustering_results(clustering, clusters, self.output_path, self.num_clusters, ref)
    def run_pipeline(self, points, vocab):
        """Run the full clustering pipeline."""
        labels, linkage_matrix = self.perform_agglomerative_clustering(points)
        clusters = {i: vocab[labels == i].tolist() for i in range(self.num_clusters)}
        self.save_clustering(labels, clusters)
        self.plot_dendrogram(linkage_matrix, 'dendrogram.png')
        return labels, clusters
# Main function for testing
def main():
    # Initialize the pipeline with default parameters
    pipeline = AgglomerativeClusteringPipeline(output_path='./output', num_clusters=5)
    # Load or generate synthetic data
    points, vocab = pipeline.load_and_prepare_data(num_points=10, num_dims=3, vocab_size=10)
    # Run the clustering pipeline
    clustering, clusters = pipeline.run_pipeline(points, vocab)
    # Plot the dendrogram for visualization
    linkage_matrix = pipeline.create_linkage_matrix(points)
    pipeline.plot_dendrogram(linkage_matrix,'dendrogram.png')
if __name__ == "__main__":
    main()
