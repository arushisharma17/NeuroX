import unittest
import numpy as np
from corpus import get_top_words

class TestGetTopWords(unittest.TestCase):

    def setUp(self):
        # Create sample tokens and activations
        self.tokens = {
            "source": [
                ["This", "is", "a", "test", "."],
                ["Another", "test", "sentence", "."],
                ["More", "data", "for", "testing", "."]
            ]
        }
        # Create activations: list of numpy arrays
        # Let's assume concatenated representation size is 3
        self.activations = [
            np.array([
                [0.1, 0.2, 0.3],  # "This"
                [0.2, 0.1, 0.0],  # "is"
                [0.3, 0.4, 0.1],  # "a"
                [0.0, 0.1, 0.2],  # "test"
                [0.1, 0.0, 0.1]   # "."
            ]),
            np.array([
                [0.2, 0.3, 0.4],  # "Another"
                [0.1, 0.2, 0.3],  # "test"
                [0.0, 0.1, 0.0],  # "sentence"
                [0.2, 0.1, 0.0]   # "."
            ]),
            np.array([
                [0.3, 0.2, 0.1],  # "More"
                [0.4, 0.3, 0.2],  # "data"
                [0.1, 0.0, 0.1],  # "for"
                [0.2, 0.1, 0.0],  # "testing"
                [0.0, 0.1, 0.2]   # "."
            ])
        ]
        self.neuron = 1  # Index of the neuron to test

    def test_get_top_words_valid(self):
        # Test with valid inputs
        top_words = get_top_words(self.tokens, self.activations, self.neuron)
        self.assertIsInstance(top_words, list)
        self.assertTrue(len(top_words) > 0)
        for word, score in top_words:
            self.assertIsInstance(word, str)
            self.assertIsInstance(score, float)
            self.assertGreaterEqual(score, 0)
            self.assertLessEqual(score, 1)

    def test_get_top_words_num_tokens(self):
        # Test specifying num_tokens
        num_tokens = 3
        top_words = get_top_words(self.tokens, self.activations, self.neuron, num_tokens=num_tokens)
        self.assertEqual(len(top_words), num_tokens)

    def test_get_top_words_min_threshold(self):
        # Test specifying min_threshold
        min_threshold = 0.5
        top_words = get_top_words(self.tokens, self.activations, self.neuron, min_threshold=min_threshold)
        for word, score in top_words:
            self.assertGreaterEqual(score, min_threshold)

    def test_get_top_words_num_tokens_and_min_threshold(self):
        # Test error when both num_tokens and min_threshold are specified
        with self.assertRaises(ValueError):
            get_top_words(self.tokens, self.activations, self.neuron, num_tokens=3, min_threshold=0.5)

    def test_get_top_words_empty_activations(self):
        # Test with empty activations
        empty_activations = []
        with self.assertRaises(ValueError):
            get_top_words(self.tokens, empty_activations, self.neuron)

    def test_get_top_words_empty_tokens(self):
        # Test with empty tokens
        empty_tokens = {"source": []}
        with self.assertRaises(ValueError):
            get_top_words(empty_tokens, self.activations, self.neuron)

    def test_get_top_words_invalid_neuron_index(self):
        # Test with invalid neuron index
        invalid_neuron = 10  # Assuming concatenated representation size is 3
        with self.assertRaises(IndexError):
            get_top_words(self.tokens, self.activations, invalid_neuron)

    def test_get_top_words_zero_std(self):
        # Test when standard deviation is zero (all activations are the same)
        activations_same = [
            np.array([
                [0.1, 0.1, 0.1],
                [0.1, 0.1, 0.1]
            ]),
            np.array([
                [0.1, 0.1, 0.1]
            ])
        ]
        with self.assertRaises(ZeroDivisionError):
            get_top_words(self.tokens, activations_same, self.neuron)

if __name__ == '__main__':
    unittest.main()
