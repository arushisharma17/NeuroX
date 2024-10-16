import unittest
from unittest.mock import patch, mock_open, MagicMock
import numpy as np
import re
from annotate import _create_binary_data, annotate_data

class TestAnnotateFunctions(unittest.TestCase):

    def setUp(self):
        # Sample tokens and activations
        self.tokens = {
            'source': [['This', 'is', 'a', 'test'], ['Another', 'test', 'sentence']],
            'target': [['O', 'O', 'O', 'O'], ['O', 'O', 'O']]
        }
        self.activations = [
            np.random.rand(4, 10),
            np.random.rand(3, 10)
        ]
        self.binary_filter_set = {'test'}
        self.binary_filter_regex = re.compile(r'^[A-Z]')
        self.binary_filter_func = lambda x: len(x) > 4

    def test_create_binary_data_with_set(self):
        words, labels, activations = _create_binary_data(
            self.tokens,
            self.activations,
            self.binary_filter_set
        )
        self.assertEqual(len(words), len(labels))
        self.assertEqual(len(words), len(activations))
        self.assertIn('positive', labels)
        self.assertIn('negative', labels)

    def test_create_binary_data_with_regex(self):
        words, labels, activations = _create_binary_data(
            self.tokens,
            self.activations,
            self.binary_filter_regex
        )
        self.assertEqual(len(words), len(labels))
        self.assertEqual(len(words), len(activations))
        self.assertIn('positive', labels)
        self.assertIn('negative', labels)

    def test_create_binary_data_with_function(self):
        words, labels, activations = _create_binary_data(
            self.tokens,
            self.activations,
            self.binary_filter_func
        )
        self.assertEqual(len(words), len(labels))
        self.assertEqual(len(words), len(activations))
        self.assertIn('positive', labels)
        self.assertIn('negative', labels)

    def test_create_binary_data_no_positive_examples(self):
        with self.assertRaises(ValueError):
            _create_binary_data(
                self.tokens,
                self.activations,
                {'nonexistentword'}
            )

    @patch('annotate.data_loader.load_activations')
    @patch('annotate.data_loader.load_data')
    @patch('annotate._create_binary_data')
    @patch('annotate.data_utils.save_files')
    def test_annotate_data(self, mock_save_files, mock_create_binary_data, mock_load_data, mock_load_activations):
        mock_load_activations.return_value = (self.activations, 1)
        mock_load_data.return_value = self.tokens
        mock_create_binary_data.return_value = (['word'], ['label'], [np.array([0.5])])
        annotate_data(
            'source.txt',
            'activations.hdf5',
            self.binary_filter_set,
            'output_prefix'
        )
        mock_load_activations.assert_called_once()
        mock_load_data.assert_called_once()
        mock_create_binary_data.assert_called_once()
        mock_save_files.assert_called_once()

if __name__ == '__main__':
    unittest.main()
