# test_utils.py

import unittest
import numpy as np
from utils import (
    isnotebook,
    get_progress_bar,
    batch_generator,
    tok2idx,
    idx2tok,
    count_target_words,
    create_tensors,
    balance_binary_class_data,
    balance_multi_class_data
)

class TestUtilsFunctions(unittest.TestCase):

    def setUp(self):
        # Sample tokens and activations
        self.tokens = {
            'source': [
                ['This', 'is', 'a', 'test', '.'],
                ['Another', 'test', 'sentence', '.']
            ],
            'target': [
                ['DET', 'VERB', 'DET', 'NOUN', 'PUNCT'],
                ['DET', 'NOUN', 'NOUN', 'PUNCT']
            ]
        }
        self.activations = [
            np.random.rand(len(self.tokens['source'][0]), 10),
            np.random.rand(len(self.tokens['source'][1]), 10)
        ]
        self.task_specific_tag = 'NOUN'
        self.mappings = None
        self.task_type = 'classification'
        self.binarized_tag = None
        self.balance_data = False

    def test_get_progress_bar(self):
        progressbar = get_progress_bar()
        self.assertIsNotNone(progressbar)

    def test_batch_generator(self):
        X = np.random.rand(100, 10)
        y = np.random.randint(0, 2, size=100)
        batch_size = 20
        generator = batch_generator(X, y, batch_size)
        batches = list(generator)
        self.assertEqual(len(batches), 5)
        for X_batch, y_batch in batches:
            self.assertEqual(X_batch.shape[0], batch_size)
            self.assertEqual(y_batch.shape[0], batch_size)

    def test_tok2idx(self):
        tokens = self.tokens['source']
        mapping = tok2idx(tokens)
        self.assertIsInstance(mapping, dict)
        self.assertTrue(all(isinstance(k, str) for k in mapping.keys()))
        self.assertTrue(all(isinstance(v, int) for v in mapping.values()))

    def test_idx2tok(self):
        tokens = self.tokens['source']
        tok2idx_mapping = tok2idx(tokens)
        idx2tok_mapping = idx2tok(tok2idx_mapping)
        self.assertIsInstance(idx2tok_mapping, dict)
        self.assertTrue(all(isinstance(k, int) for k in idx2tok_mapping.keys()))
        self.assertTrue(all(isinstance(v, str) for v in idx2tok_mapping.values()))

    def test_count_target_words(self):
        count = count_target_words(self.tokens)
        expected_count = sum(len(s) for s in self.tokens['target'])
        self.assertEqual(count, expected_count)

    def test_create_tensors(self):
        X, y, mappings = create_tensors(
            self.tokens,
            self.activations,
            self.task_specific_tag,
            task_type=self.task_type,
            binarized_tag=self.binarized_tag,
            balance_data=self.balance_data
        )
        self.assertIsInstance(X, np.ndarray)
        self.assertIsInstance(y, np.ndarray)
        self.assertEqual(X.shape[0], y.shape[0])
        if self.task_type == 'classification':
            self.assertEqual(len(mappings), 4)  # label2idx, idx2label, src2idx, idx2src
        else:
            self.assertEqual(len(mappings), 2)  # src2idx, idx2src

    def test_balance_binary_class_data(self):
        X = np.random.rand(100, 10)
        y = np.array([0]*80 + [1]*20)
        X_balanced, y_balanced = balance_binary_class_data(X, y)
        unique, counts = np.unique(y_balanced, return_counts=True)
        self.assertEqual(len(unique), 2)
        self.assertEqual(counts[0], counts[1])

    def test_balance_multi_class_data(self):
        X = np.random.rand(150, 10)
        y = np.array([0]*70 + [1]*50 + [2]*30)
        X_balanced, y_balanced = balance_multi_class_data(X, y)
        unique, counts = np.unique(y_balanced, return_counts=True)
        min_count = min(counts)
        self.assertTrue(all(count == min_count for count in counts))

    def test_balance_multi_class_data_with_num_required_instances(self):
        X = np.random.rand(150, 10)
        y = np.array([0]*70 + [1]*50 + [2]*30)
        num_required_instances = 60
        X_balanced, y_balanced = balance_multi_class_data(X, y, num_required_instances)
        self.assertEqual(len(y_balanced), num_required_instances)
        unique, counts = np.unique(y_balanced, return_counts=True)
        total_counts = sum(counts)
        self.assertEqual(total_counts, num_required_instances)

if __name__ == '__main__':
    unittest.main()
