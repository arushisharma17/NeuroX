import unittest
import numpy as np
from unittest.mock import patch, MagicMock, mock_open
from utils import (
    _balance_negative_class,
    save_files
)
from neurox.data.writer import ActivationsWriter

class TestUtilsFunctions(unittest.TestCase):

    def setUp(self):
        # Sample data for testing
        self.words = ['word1', 'word2', 'word3', 'word4', 'word5']
        self.activations = [np.random.rand(5, 10) for _ in range(len(self.words))]
        self.positive_class_size = 3
        self.output_prefix = 'test_output'
        self.labels = ['label1', 'label2', 'label3', 'label4', 'label5']

    def test_balance_negative_class_smaller_negative_class(self):
        words = ['word1', 'word2']
        activations = [np.random.rand(5, 10), np.random.rand(5, 10)]
        with patch('builtins.print') as mock_print:
            balanced_words, balanced_activations = _balance_negative_class(
                words, activations, self.positive_class_size
            )
            mock_print.assert_any_call(
                'No need of balancing the data. Negative class is equal or smaller in size to the positive class'
            )
        self.assertEqual(balanced_words, words)
        self.assertEqual(balanced_activations, activations)

    def test_balance_negative_class_balancing(self):
        words = ['word{}'.format(i) for i in range(10)]
        activations = [np.random.rand(5, 10) for _ in range(len(words))]
        balanced_words, balanced_activations = _balance_negative_class(
            words, activations, self.positive_class_size
        )
        self.assertEqual(len(balanced_words), self.positive_class_size)
        self.assertEqual(len(balanced_activations), self.positive_class_size)

    @patch('builtins.open', new_callable=mock_open)
    @patch('utils.ActivationsWriter.get_writer')
    def test_save_files(self, mock_get_writer, mock_file):
        mock_writer_instance = MagicMock()
        mock_get_writer.return_value = mock_writer_instance

        save_files(
            self.words,
            self.labels,
            self.activations,
            self.output_prefix
        )

        # Check that word file is written correctly
        mock_file.assert_any_call(self.output_prefix + '.word', 'w')
        word_file_handle = mock_file()
        word_file_handle.write.assert_called()

        # Check that label file is written correctly
        mock_file.assert_any_call(self.output_prefix + '.label', 'w')
        label_file_handle = mock_file()
        label_file_handle.write.assert_called()

        # Check that activations are written using ActivationsWriter
        self.assertTrue(mock_get_writer.called)
        self.assertTrue(mock_writer_instance.write_activations.called)
        self.assertTrue(mock_writer_instance.close.called)

if __name__ == '__main__':
    unittest.main()
