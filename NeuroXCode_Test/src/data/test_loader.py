import unittest
import numpy as np
import os
from unittest.mock import mock_open, patch
from loader import (
    load_activations,
    filter_activations_by_layers,
    load_aux_data,
    load_data,
    load_sentence_data
)

class TestLoaderFunctions(unittest.TestCase):

    def setUp(self):
        # Sample activations and tokens
        self.activations_path = 'test_activations.hdf5'
        self.activations_data = [
            np.random.rand(4, 10)
        ]
        self.num_layers = 2
        self.num_neurons_per_layer = 5
        self.is_brnn = False

        # Create sample source and labels files
        self.source_content = 'This is a test\nAnother test sentence\n'
        self.labels_content = 'DET VERB DET NOUN\nDET NOUN NOUN\n'
        self.source_path = 'test_source.txt'
        self.labels_path = 'test_labels.txt'
        with open(self.source_path, 'w') as f:
            f.write(self.source_content)
        with open(self.labels_path, 'w') as f:
            f.write(self.labels_content)

    def tearDown(self):
        # Remove temporary files
        if os.path.exists(self.source_path):
            os.remove(self.source_path)
        if os.path.exists(self.labels_path):
            os.remove(self.labels_path)

    @patch('loader.h5py.File')
    def test_load_activations_hdf5(self, mock_h5py_file):
        # Mock h5py.File
        mock_file = mock_h5py_file.return_value
        mock_file.get.return_value = ['{"sentence": "0"}']
        mock_file.__getitem__.return_value = np.random.rand(2, 4, 5)
        activations, num_layers = load_activations(self.activations_path)
        self.assertIsInstance(activations, list)
        self.assertIsInstance(activations[0], np.ndarray)
        self.assertEqual(num_layers, 1)

    def test_load_data(self):
        activations = [np.random.rand(4, 10), np.random.rand(3, 10)]
        tokens = load_data(
            self.source_path,
            self.labels_path,
            activations,
            max_sent_l=10
        )
        self.assertIn('source', tokens)
        self.assertIn('target', tokens)
        self.assertEqual(len(tokens['source']), 2)
        self.assertEqual(len(tokens['target']), 2)

    def test_load_sentence_data(self):
        activations = [np.random.rand(1, 10), np.random.rand(1, 10)]
        tokens = load_sentence_data(
            self.source_path,
            self.labels_path,
            activations
        )
        self.assertIn('source', tokens)
        self.assertIn('target', tokens)
        self.assertEqual(len(tokens['source']), 2)
        self.assertEqual(len(tokens['target']), 2)

    def test_invalid_activations_length(self):
        activations = [np.random.rand(4, 10)]  # Only one activation
        with self.assertRaises(AssertionError):
            load_data(
                self.source_path,
                self.labels_path,
                activations,
                max_sent_l=10
            )

    def test_load_activations_invalid_file(self):
        with self.assertRaises(AssertionError):
            load_activations('activations.invalid')

if __name__ == '__main__':
    unittest.main()
