import unittest
import numpy as np
from clustering import (
    create_correlation_clusters,
    extract_independent_neurons,
    print_clusters,
    scikit_extract_independent_neurons,
)

class TestClusteringFunctions(unittest.TestCase):

    def setUp(self):
        # Create a sample activation matrix X
        self.num_tokens = 50
        self.num_neurons = 10
        np.random.seed(42)
        self.X = np.random.rand(self.num_tokens, self.num_neurons)

    def test_create_correlation_clusters(self):
        labels = create_correlation_clusters(self.X)
        self.assertEqual(len(labels), self.num_neurons)
        self.assertTrue(np.max(labels) > 0)
        self.assertTrue(np.min(labels) > 0)

    def test_extract_independent_neurons(self):
        independent_neurons, clusters = extract_independent_neurons(self.X)
        self.assertIsInstance(independent_neurons, list)
        self.assertEqual(len(independent_neurons), np.max(clusters))
        self.assertTrue(all(0 <= n < self.num_neurons for n in independent_neurons))

    def test_print_clusters(self):
        # Capture print output
        import io
        import sys
        captured_output = io.StringIO()
        sys.stdout = captured_output
        labels = create_correlation_clusters(self.X)
        print_clusters(labels)
        sys.stdout = sys.__stdout__
        output = captured_output.getvalue()
        self.assertTrue("Cluster" in output)

    def test_scikit_extract_independent_neurons(self):
        independent_neurons, clusters = scikit_extract_independent_neurons(self.X)
        self.assertIsInstance(independent_neurons, list)
        self.assertEqual(len(independent_neurons), np.max(clusters))
        self.assertTrue(all(0 <= n < self.num_neurons for n in independent_neurons))

    def test_compare_methods(self):
        # Ensure both methods give the same results
        ind_neurons1, clusters1 = extract_independent_neurons(self.X)
        ind_neurons2, clusters2 = scikit_extract_independent_neurons(self.X)
        np.testing.assert_array_equal(clusters1, clusters2)

if __name__ == '__main__':
    unittest.main()
