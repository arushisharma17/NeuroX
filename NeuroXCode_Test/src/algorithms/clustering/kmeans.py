import unittest
import numpy as np
from kmeans import KMeansClusteringPipeline

class TestKMeansClusteringPipeline(unittest.TestCase):

    def setUp(self):
        # Initialize the pipeline with test parameters
        self.pipeline = KMeansClusteringPipeline(num_clusters=3)
        # Generate synthetic data for testing
        self.num_points = 10
        self.num_dims = 3
        self.vocab_size = 10
        self.points = np.random.rand(self.num_points, self.num_dims)
        self.vocab = np.array([f"word_{i}" for i in range(self.vocab_size)])

    def test_load_and_prepare_data(self):
        # Test data loading and preparation
        points, vocab = self.pipeline.load_and_prepare_data(num_points=self.num_points, num_dims=self.num_dims, vocab_size=self.vocab_size)
        self.assertEqual(points.shape, (self.num_points, self.num_dims))
        self.assertEqual(len(vocab), self.vocab_size)

    def test_perform_kmeans_clustering(self):
        # Test the clustering method
        clustering = self.pipeline.perform_kmeans_clustering(self.points)
        self.assertEqual(len(clustering.labels_), self.num_points)
        self.assertEqual(len(set(clustering.labels_)), self.pipeline.num_clusters)

    def test_run_pipeline(self):
        # Test the full pipeline
        clustering, clusters, duration = self.pipeline.run_pipeline(self.points, self.vocab)
        self.assertEqual(len(clustering.labels_), self.num_points)
        self.assertEqual(len(clusters), self.pipeline.num_clusters)
        self.assertIsInstance(duration, float)
        for cluster_id, words in clusters.items():
            self.assertIsInstance(words, list)
            self.assertGreater(len(words), 0)

if __name__ == "__main__":
    unittest.main()
