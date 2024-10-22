import unittest
import numpy as np
from probeless import (
    get_neuron_ordering,
    get_neuron_ordering_for_tag,
    get_neuron_ordering_for_all_tags
)

class TestProbelessMethods(unittest.TestCase):

    def setUp(self):
        # Create synthetic data
        self.num_samples = 100
        self.num_neurons = 20
        self.num_classes = 3
        np.random.seed(42)

        self.X_train = np.random.rand(self.num_samples, self.num_neurons)
        self.y_train = np.random.randint(0, self.num_classes, size=self.num_samples)

        self.label2idx = {f'class_{i}': i for i in range(self.num_classes)}
        self.idx2label = {i: f'class_{i}' for i in range(self.num_classes)}
        self.tag = 'class_1'

    def test_get_neuron_ordering(self):
        ranking = get_neuron_ordering(self.X_train, self.y_train)
        self.assertEqual(len(ranking), self.num_neurons)
        self.assertTrue(all(0 <= idx < self.num_neurons for idx in ranking))

    def test_get_neuron_ordering_for_tag(self):
        ranking = get_neuron_ordering_for_tag(
            self.X_train,
            self.y_train,
            self.label2idx,
            self.tag
        )
        self.assertEqual(len(ranking), self.num_neurons)
        self.assertTrue(all(0 <= idx < self.num_neurons for idx in ranking))

    def test_get_neuron_ordering_for_all_tags(self):
        overall_ranking, ranking_per_tag = get_neuron_ordering_for_all_tags(
            self.X_train,
            self.y_train,
            self.idx2label
        )
        self.assertEqual(len(overall_ranking), self.num_neurons)
        self.assertEqual(len(ranking_per_tag), self.num_classes)
        for tag, ranking in ranking_per_tag.items():
            self.assertEqual(len(ranking), self.num_neurons)
            self.assertTrue(all(0 <= idx < self.num_neurons for idx in ranking))

    def test_invalid_tag(self):
        with self.assertRaises(KeyError):
            get_neuron_ordering_for_tag(
                self.X_train,
                self.y_train,
                self.label2idx,
                'invalid_tag'
            )

if __name__ == '__main__':
    unittest.main()
