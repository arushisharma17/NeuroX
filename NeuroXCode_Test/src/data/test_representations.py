import unittest
import numpy as np
from representations import (
    bpe_get_avg_activations,
    bpe_get_last_activations,
    char_get_avg_activations,
    char_get_last_activations,
    sent_get_last_activations
)
from unittest.mock import patch

class TestRepresentationsFunctions(unittest.TestCase):

    def setUp(self):
        # Sample tokens and activations
        self.tokens = {
            'source': [['This', 'is', 'a', 'test']],
            'source_aux': [['Th@@', 'is', 'i@@', 's', 'a', 'te@@', 'st']],
            'target': [['DET', 'VERB', 'DET', 'NOUN']]
        }
        self.activations = [
            np.random.rand(len(self.tokens['source_aux'][0]), 10)
        ]

    @patch('representations.tqdm', lambda x: x)  # Mock tqdm for testing
    def test_bpe_get_avg_activations(self):
        aggregated_activations = bpe_get_avg_activations(self.tokens, self.activations)
        self.assertEqual(len(aggregated_activations), len(self.tokens['source']))
        self.assertEqual(aggregated_activations[0].shape[0], len(self.tokens['source'][0]))
        self.assertEqual(aggregated_activations[0].shape[1], 10)

    @patch('representations.tqdm', lambda x: x)
    def test_bpe_get_last_activations(self):
        aggregated_activations = bpe_get_last_activations(self.tokens, self.activations)
        self.assertEqual(len(aggregated_activations), len(self.tokens['source']))
        self.assertEqual(aggregated_activations[0].shape[0], len(self.tokens['source'][0]))
        self.assertEqual(aggregated_activations[0].shape[1], 10)

    @patch('representations.tqdm', lambda x: x)
    def test_char_get_avg_activations(self):
        # Adjust tokens to simulate character-level tokenization
        self.tokens['source_aux'] = [['T', 'h', 'i', 's', '_', 'i', 's', '_', 'a', '_', 't', 'e', 's', 't']]
        self.activations = [
            np.random.rand(len(self.tokens['source_aux'][0]), 10)
        ]
        aggregated_activations = char_get_avg_activations(self.tokens, self.activations)
        self.assertEqual(len(aggregated_activations), len(self.tokens['source']))
        self.assertEqual(aggregated_activations[0].shape[0], len(self.tokens['source'][0]))
        self.assertEqual(aggregated_activations[0].shape[1], 10)

    @patch('representations.tqdm', lambda x: x)
    def test_char_get_last_activations(self):
        # Adjust tokens to simulate character-level tokenization
        self.tokens['source_aux'] = [['T', 'h', 'i', 's', '_', 'i', 's', '_', 'a', '_', 't', 'e', 's', 't']]
        self.activations = [
            np.random.rand(len(self.tokens['source_aux'][0]), 10)
        ]
        aggregated_activations = char_get_last_activations(self.tokens, self.activations)
        self.assertEqual(len(aggregated_activations), len(self.tokens['source']))
        self.assertEqual(aggregated_activations[0].shape[0], len(self.tokens['source'][0]))
        self.assertEqual(aggregated_activations[0].shape[1], 10)

    @patch('representations.tqdm', lambda x: x)
    def test_sent_get_last_activations(self):
        activations = [
            np.random.rand(len(self.tokens['source'][0]), 10)
        ]
        sentence_activations = sent_get_last_activations(self.tokens, activations)
        self.assertEqual(len(sentence_activations), len(self.tokens['source']))
        self.assertEqual(sentence_activations[0].shape[0], 1)
        self.assertEqual(sentence_activations[0].shape[1], 10)

if __name__ == '__main__':
    unittest.main()
