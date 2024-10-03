import unittest
import numpy as np
from leaders import LeadersClusteringPipeline

class TestLeadersClusteringPipeline(unittest.TestCase):

    def setUp(self):
        # Initialize the pipeline with test parameters
        self.pipeline = LeadersClusteringPipeline(K=3, tau=0.5, is_fast=True)
        # Generate synthetic data for testing
        self.num_points = 20
        self.num_dims = 5
        self.vocab_size = 20
        self.points = np.random.rand(self.num_points, self.num_dims)
        self.vocab = np.array([f"word_{i}" for i in range(self.vocab_size)])

    def test_calculate_tau(self):
        # Test tau calculation method
        from annoy import AnnoyIndex
        t = AnnoyIndex(self.num_dims, 'euclidean')
        for i, p in enumerate(self.points):
            t.add_item(i, p)
        t.build(10)
        tau = self.pipeline.calculate_tau(t, self.points)
        self.assertIsInstance(tau, float)
        self.assertGreater(tau, 0)

    def test_leaders_cluster(self):
        # Test the leaders clustering algorithm
        clustering, clusters = self.pipeline.leaders_cluster(self.points, self.vocab)
        self.assertIsInstance(clusters, dict)
        self.assertLessEqual(len(clusters), self.pipeline.K)
        for cluster_id, words in clusters.items():
            self.assertIsInstance(words, list)
            self.assertGreater(len(words), 0)

    def test_run_pipeline(self):
        # Test the full pipeline
        clustering, clusters = self.pipeline.run_pipeline(self.points, self.vocab)
        self.assertIsInstance(clusters, dict)
        self.assertLessEqual(len(clusters), self.pipeline.K)
        for cluster_id, words in clusters.items():
            self.assertIsInstance(words, list)
            self.assertGreater(len(words), 0)

if __name__ == "__main__":
    unittest.main()
