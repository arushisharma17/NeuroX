import unittest
import numpy as np
from gaussian_probe import GaussianProbe, train_probe, evaluate_probe, get_neuron_ordering

class TestGaussianProbe(unittest.TestCase):

    def setUp(self):
        # Create a synthetic dataset
        self.num_samples = 100
        self.num_neurons = 10
        self.num_classes = 3
        np.random.seed(42)
        self.X = np.random.rand(self.num_samples, self.num_neurons)
        self.y = np.random.randint(0, self.num_classes, size=self.num_samples)

    def test_train_probe(self):
        probe = train_probe(self.X, self.y)
        self.assertIsInstance(probe, GaussianProbe)

    def test_evaluate_probe(self):
        probe = train_probe(self.X, self.y)
        result = evaluate_probe(probe, self.X, self.y)
        self.assertIn('__OVERALL__', result)
        self.assertTrue(0.0 <= result['__OVERALL__'] <= 1.0)

    def test_evaluate_probe_with_predictions(self):
        probe = train_probe(self.X, self.y)
        preds, result = evaluate_probe(probe, self.X, self.y, return_predictions=True)
        self.assertEqual(len(preds), self.num_samples)
        self.assertIn('__OVERALL__', result)

    def test_get_neuron_ordering(self):
        probe = train_probe(self.X, self.y)
        num_of_neurons = 5
        neuron_ordering = get_neuron_ordering(probe, num_of_neurons)
        self.assertEqual(len(neuron_ordering), num_of_neurons)
        self.assertTrue(all(0 <= n < self.num_neurons for n in neuron_ordering))

    def test_invalid_num_of_neurons(self):
        probe = train_probe(self.X, self.y)
        num_of_neurons = 500  # Exceeds num_neurons
        with self.assertRaises(Exception):
            get_neuron_ordering(probe, num_of_neurons)

    def test_evaluate_probe_with_selected_neurons(self):
        probe = train_probe(self.X, self.y)
        selected_neurons = [0, 1, 2]
        result = evaluate_probe(probe, self.X, self.y, selected_neurons=selected_neurons)
        self.assertIn('__OVERALL__', result)

if __name__ == '__main__':
    unittest.main()
