import unittest
import numpy as np
from iou_probe import get_neuron_ordering

class TestIoUProbe(unittest.TestCase):

    def setUp(self):
        # Create synthetic data
        self.num_samples = 100
        self.num_neurons = 20
        np.random.seed(42)
        self.X_train = np.random.rand(self.num_samples, self.num_neurons)
        self.y_train = np.random.randint(0, 2, size=self.num_samples)
        self.threshold = 0.05

    def test_get_neuron_ordering(self):
        ranking = get_neuron_ordering(self.X_train, self.y_train, self.threshold)
        self.assertEqual(len(ranking), self.num_neurons)
        self.assertTrue(all(0 <= idx < self.num_neurons for idx in ranking))

    def test_get_neuron_ordering_different_threshold(self):
        threshold = 0.1
        ranking = get_neuron_ordering(self.X_train, self.y_train, threshold)
        self.assertEqual(len(ranking), self.num_neurons)

    def test_get_neuron_ordering_all_zero_activations(self):
        X_zero = np.zeros_like(self.X_train)
        ranking = get_neuron_ordering(X_zero, self.y_train, self.threshold)
        self.assertEqual(len(ranking), self.num_neurons)
        # Since all activations are zero, the scores should be NaN
        # Average precision score will be undefined
        # We can check for warnings or handle this case

    def test_get_neuron_ordering_invalid_inputs(self):
        with self.assertRaises(ValueError):
            get_neuron_ordering(self.X_train, self.y_train[:50], self.threshold)

    def test_get_neuron_ordering_non_binary_labels(self):
        y_multi = np.random.randint(0, 3, size=self.num_samples)
        with self.assertRaises(ValueError):
            get_neuron_ordering(self.X_train, y_multi, self.threshold)

if __name__ == '__main__':
    unittest.main()
