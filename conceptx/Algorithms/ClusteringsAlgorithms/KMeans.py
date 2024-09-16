import os
from sklearn.cluster import KMeans
from conceptx.Utilities.utils import load_data, save_clustering_results, log_clustering_process
import time
import logging

class KMeansClusteringPipeline:
    def __init__(self, output_path='./output', num_clusters=5):
        self.output_path = output_path
        self.num_clusters = num_clusters
        os.makedirs(self.output_path, exist_ok=True)

    def load_and_prepare_data(self, point_file=None, vocab_file=None, num_points=100, num_dims=5, vocab_size=100):
        """Load or generate synthetic data."""
        points, vocab = load_data(point_file, vocab_file, num_points, num_dims, vocab_size, self.output_path)
        return points, vocab

    def perform_kmeans_clustering(self, data):
        """Perform K-Means clustering on the input data."""
        kmeans = KMeans(n_clusters=self.num_clusters, verbose=3)
        kmeans.fit(data)
        return kmeans

    def save_clustering(self, clustering, clusters, ref=''):
        """Save the K-Means clustering results."""
        save_clustering_results(clustering, clusters, self.output_path, self.num_clusters, ref)

    def run_pipeline(self, points, vocab):
        """Run the full K-Means clustering pipeline."""
        start_time = time.time()
        clustering = self.perform_kmeans_clustering(points)
        clusters = {i: vocab[clustering.labels_ == i].tolist() for i in range(self.num_clusters)}
        self.save_clustering(clustering, clusters)
        end_time = time.time()
        log_clustering_process(points, vocab, self.num_clusters, start_time, end_time, clusters)
        return clustering, clusters

# Main function for testing
def main():
    # Set up logging
    output_path = '../../output'
    os.makedirs(output_path, exist_ok=True)
    logging.basicConfig(filename=os.path.join(output_path, 'kmeans_clustering.log'),
                        level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

    # Initialize the pipeline with default parameters
    pipeline = KMeansClusteringPipeline(output_path=output_path, num_clusters=5)

    # Load or generate synthetic data
    points, vocab = pipeline.load_and_prepare_data(num_points=10, num_dims=3, vocab_size=10)

    # Run the clustering pipeline
    clustering, clusters = pipeline.run_pipeline(points, vocab)

if __name__ == "__main__":
    main()