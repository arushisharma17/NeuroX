import unittest
import numpy as np
import warnings
from ablation import (
    keep_specific_neurons,
    filter_activations_keep_neurons,
    filter_activations_remove_neurons,
    zero_out_activations_keep_neurons,
    zero_out_activations_remove_neurons,
    filter_activations_by_layers,
)

class TestAblationFunctions(unittest.TestCase):

    def setUp(self):
        # Create a sample activation matrix X
        self.num_tokens = 10
        self.num_neurons = 20
        self.X = np.random.rand(self.num_tokens, self.num_neurons)
        self.neurons_to_keep = [0, 2, 4, 6, 8]
        self.neurons_to_remove = [1, 3, 5, 7, 9]
        self.layers_to_keep = [0, 1]
        self.num_layers = 5  # Assume 5 layers
        self.bidirectional_filtering = 'none'

    def test_keep_specific_neurons(self):
        # Test deprecated function with warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            filtered_X = keep_specific_neurons(self.X, self.neurons_to_keep)
            self.assertEqual(filtered_X.shape, (self.num_tokens, len(self.neurons_to_keep)))
            # Check if warning was raised
            self.assertTrue(len(w) > 0)
            self.assertTrue(issubclass(w[-1].category, DeprecationWarning))

    def test_filter_activations_keep_neurons(self):
        filtered_X = filter_activations_keep_neurons(self.X, self.neurons_to_keep)
        self.assertEqual(filtered_X.shape, (self.num_tokens, len(self.neurons_to_keep)))
        np.testing.assert_array_equal(filtered_X, self.X[:, self.neurons_to_keep])

    def test_filter_activations_remove_neurons(self):
        filtered_X = filter_activations_remove_neurons(self.X, self.neurons_to_remove)
        expected_neurons = [i for i in range(self.num_neurons) if i not in self.neurons_to_remove]
        self.assertEqual(filtered_X.shape, (self.num_tokens, len(expected_neurons)))
        np.testing.assert_array_equal(filtered_X, self.X[:, expected_neurons])

    def test_zero_out_activations_keep_neurons(self):
        zeroed_X = zero_out_activations_keep_neurons(self.X, self.neurons_to_keep)
        self.assertEqual(zeroed_X.shape, self.X.shape)
        # Check that only neurons_to_keep are non-zero
        non_zero_indices = np.nonzero(zeroed_X)
        self.assertTrue(all(idx in self.neurons_to_keep for idx in non_zero_indices[1]))

    def test_zero_out_activations_remove_neurons(self):
        zeroed_X = zero_out_activations_remove_neurons(self.X, self.neurons_to_remove)
        self.assertEqual(zeroed_X.shape, self.X.shape)
        # Check that neurons_to_remove are zeroed out
        for neuron in self.neurons_to_remove:
            self.assertTrue(np.all(zeroed_X[:, neuron] == 0))
        # Check that other neurons are unchanged
        for neuron in range(self.num_neurons):
            if neuron not in self.neurons_to_remove:
                np.testing.assert_array_equal(zeroed_X[:, neuron], self.X[:, neuron])

    def test_filter_activations_by_layers(self):
        filtered_X = filter_activations_by_layers(
            self.X,
            self.layers_to_keep,
            self.num_layers,
            self.bidirectional_filtering
        )
        neurons_per_layer = self.X.shape[1] // self.num_layers
        expected_neurons = []
        for layer in self.layers_to_keep:
            start = layer * neurons_per_layer
            end = start + neurons_per_layer
            expected_neurons.extend(range(start, end))
        self.assertEqual(filtered_X.shape, (self.num_tokens, len(expected_neurons)))
        np.testing.assert_array_equal(filtered_X, self.X[:, expected_neurons])

    def test_filter_activations_by_layers_bidirectional(self):
        # Test with bidirectional filtering
        bidirectional_filtering = 'forward'
        filtered_X = filter_activations_by_layers(
            self.X,
            self.layers_to_keep,
            self.num_layers,
            bidirectional_filtering
        )
        neurons_per_layer = self.X.shape[1] // (self.num_layers * 2)
        expected_neurons = []
        for layer in self.layers_to_keep:
            start = layer * (neurons_per_layer * 2)
            end = start + neurons_per_layer
            expected_neurons.extend(range(start, end))
        self.assertEqual(filtered_X.shape, (self.num_tokens, len(expected_neurons)))
        np.testing.assert_array_equal(filtered_X, self.X[:, expected_neurons])

    def test_filter_activations_by_layers_invalid_direction(self):
        with self.assertRaises(AssertionError):
            filter_activations_by_layers(
                self.X,
                self.layers_to_keep,
                self.num_layers,
                bidirectional_filtering="invalid"
            )

if __name__ == '__main__':
    unittest.main()
