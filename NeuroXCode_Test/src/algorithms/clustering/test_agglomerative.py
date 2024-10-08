import unittest
import numpy as np
import os
from agglomerative import AgglomerativeClusteringPipeline

class TestAgglomerativeClusteringPipeline(unittest.TestCase):

    def setUp(self):
        # Initialize the pipeline with test parameters
        self.pipeline = AgglomerativeClusteringPipeline(output_path='./test_output', num_clusters=3)
        os.makedirs(self.pipeline.output_path, exist_ok=True)
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

    def test_perform_agglomerative_clustering(self):
        # Test the clustering method
        clustering = self.pipeline.perform_agglomerative_clustering(self.points)
        self.assertEqual(len(clustering.labels_), self.num_points)
        self.assertEqual(len(set(clustering.labels_)), self.pipeline.num_clusters)

    def test_create_linkage_matrix(self):
        # Test linkage matrix creation
        linkage_matrix = self.pipeline.create_linkage_matrix(self.points)
        self.assertEqual(linkage_matrix.shape, (self.num_points - 1, 4))

    def test_plot_dendrogram(self):
        # Test dendrogram plotting
        linkage_matrix = self.pipeline.create_linkage_matrix(self.points)
        self.pipeline.plot_dendrogram(linkage_matrix, 'test_dendrogram.png')
        self.assertTrue(os.path.exists(f"{self.pipeline.output_path}/test_dendrogram.png"))

    def test_save_clustering(self):
        # Test saving clustering results
        clustering = self.pipeline.perform_agglomerative_clustering(self.points)
        clusters = {i: self.vocab[clustering.labels_ == i].tolist() for i in range(self.pipeline.num_clusters)}
        self.pipeline.save_clustering(clustering, clusters, ref='test')
        expected_file = f"{self.pipeline.output_path}/clustering_results_test.txt"
        self.assertTrue(os.path.exists(expected_file))

    def test_run_pipeline(self):
        # Test the full pipeline
        clustering, clusters = self.pipeline.run_pipeline(self.points, self.vocab)
        self.assertEqual(len(clustering.labels_), self.num_points)
        self.assertEqual(len(clusters), self.pipeline.num_clusters)
        for cluster_id, words in clusters.items():
            self.assertIsInstance(words, list)
            self.assertGreater(len(words), 0)

    def tearDown(self):
        # Clean up test output directory
        if os.path.exists(self.pipeline.output_path):
            for file in os.listdir(self.pipeline.output_path):
                os.remove(os.path.join(self.pipeline.output_path, file))
            os.rmdir(self.pipeline.output_path)

if __name__ == "__main__":
    unittest.main()
