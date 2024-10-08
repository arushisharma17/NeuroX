import unittest
import numpy as np
import torch
from linear_probe import (
    LinearProbe,
    train_logistic_regression_probe,
    train_linear_regression_probe,
    evaluate_probe,
    get_top_neurons,
    get_bottom_neurons,
    get_random_neurons,
    get_neuron_ordering,
    get_neuron_ordering_granular,
)

class TestLinearProbe(unittest.TestCase):

    def setUp(self):
        # Create synthetic data
        self.num_samples = 100
        self.num_neurons = 20
        self.num_classes = 3
        np.random.seed(42)
        torch.manual_seed(42)

        # Classification data
        self.X_classification = np.random.rand(self.num_samples, self.num_neurons).astype(np.float32)
        self.y_classification = np.random.randint(0, self.num_classes, size=self.num_samples)

        # Regression data
        self.X_regression = np.random.rand(self.num_samples, self.num_neurons).astype(np.float32)
        self.y_regression = np.random.rand(self.num_samples).astype(np.float32)

        # Class to index mapping
        self.class_to_idx = {str(i): i for i in range(self.num_classes)}
        self.idx_to_class = {i: str(i) for i in range(self.num_classes)}

    def test_linear_probe_initialization(self):
        input_size = self.num_neurons
        num_classes = self.num_classes
        probe = LinearProbe(input_size, num_classes)
        self.assertIsInstance(probe, LinearProbe)
        self.assertEqual(probe.linear.in_features, input_size)
        self.assertEqual(probe.linear.out_features, num_classes)

    def test_train_logistic_regression_probe(self):
        probe = train_logistic_regression_probe(
            self.X_classification,
            self.y_classification,
            num_epochs=5,
            batch_size=16,
            learning_rate=0.01
        )
        self.assertIsInstance(probe, LinearProbe)

    def test_train_linear_regression_probe(self):
        probe = train_linear_regression_probe(
            self.X_regression,
            self.y_regression,
            num_epochs=5,
            batch_size=16,
            learning_rate=0.01
        )
        self.assertIsInstance(probe, LinearProbe)

    def test_evaluate_probe_classification(self):
        probe = train_logistic_regression_probe(
            self.X_classification,
            self.y_classification,
            num_epochs=5,
            batch_size=16,
            learning_rate=0.01
        )
        scores = evaluate_probe(
            probe,
            self.X_classification,
            self.y_classification,
            idx_to_class=self.idx_to_class,
            batch_size=16,
            metric="accuracy"
        )
        self.assertIn('__OVERALL__', scores)
        self.assertTrue(0.0 <= scores['__OVERALL__'] <= 1.0)
        # Check per-class scores
        for class_name in self.class_to_idx.keys():
            self.assertIn(class_name, scores)
            self.assertTrue(0.0 <= scores[class_name] <= 1.0)

    def test_evaluate_probe_regression(self):
        probe = train_linear_regression_probe(
            self.X_regression,
            self.y_regression,
            num_epochs=5,
            batch_size=16,
            learning_rate=0.01
        )
        scores = evaluate_probe(
            probe,
            self.X_regression,
            self.y_regression,
            batch_size=16,
            metric="pearson"
        )
        self.assertIn('__OVERALL__', scores)
        # Pearson correlation can be between -1 and 1
        self.assertTrue(-1.0 <= scores['__OVERALL__'] <= 1.0)

    def test_get_top_neurons(self):
        probe = train_logistic_regression_probe(
            self.X_classification,
            self.y_classification,
            num_epochs=5,
            batch_size=16,
            learning_rate=0.01
        )
        percentage = 0.1
        top_neurons, top_neurons_per_class = get_top_neurons(
            probe,
            percentage,
            self.class_to_idx
        )
        self.assertTrue(len(top_neurons) <= self.num_neurons)
        self.assertTrue(len(top_neurons) > 0)
        # Check top neurons per class
        for class_name in self.class_to_idx.keys():
            self.assertIn(class_name, top_neurons_per_class)
            self.assertIsInstance(top_neurons_per_class[class_name], np.ndarray)

    def test_get_bottom_neurons(self):
        probe = train_logistic_regression_probe(
            self.X_classification,
            self.y_classification,
            num_epochs=5,
            batch_size=16,
            learning_rate=0.01
        )
        percentage = 0.1
        bottom_neurons, bottom_neurons_per_class = get_bottom_neurons(
            probe,
            percentage,
            self.class_to_idx
        )
        self.assertTrue(len(bottom_neurons) <= self.num_neurons)
        self.assertTrue(len(bottom_neurons) > 0)
        # Check bottom neurons per class
        for class_name in self.class_to_idx.keys():
            self.assertIn(class_name, bottom_neurons_per_class)
            self.assertIsInstance(bottom_neurons_per_class[class_name], np.ndarray)

    def test_get_random_neurons(self):
        probe = train_logistic_regression_probe(
            self.X_classification,
            self.y_classification,
            num_epochs=5,
            batch_size=16,
            learning_rate=0.01
        )
        probability = 0.1
        random_neurons = get_random_neurons(probe, probability)
        self.assertTrue(len(random_neurons) <= self.num_neurons)
        self.assertTrue(len(random_neurons) > 0)

    def test_get_neuron_ordering(self):
        probe = train_logistic_regression_probe(
            self.X_classification,
            self.y_classification,
            num_epochs=5,
            batch_size=16,
            learning_rate=0.01
        )
        ordering, cutoffs = get_neuron_ordering(
            probe,
            self.class_to_idx,
            search_stride=10
        )
        self.assertEqual(len(ordering), self.num_neurons)
        self.assertIsInstance(cutoffs, list)

    def test_get_neuron_ordering_granular(self):
        probe = train_logistic_regression_probe(
            self.X_classification,
            self.y_classification,
            num_epochs=5,
            batch_size=16,
            learning_rate=0.01
        )
        ordering, cutoffs = get_neuron_ordering_granular(
            probe,
            self.class_to_idx,
            granularity=5,
            search_stride=10
        )
        self.assertEqual(len(ordering), self.num_neurons)
        self.assertIsInstance(cutoffs, list)

    def test_invalid_task_type(self):
        with self.assertRaises(ValueError):
            _ = train_logistic_regression_probe(
                self.X_classification,
                self.y_classification,
                num_epochs=5,
                batch_size=16,
                learning_rate=0.01,
                task_type='invalid'
            )

    def test_evaluate_probe_invalid_metric(self):
        probe = train_logistic_regression_probe(
            self.X_classification,
            self.y_classification,
            num_epochs=1
        )
        with self.assertRaises(ValueError):
            evaluate_probe(
                probe,
                self.X_classification,
                self.y_classification,
                metric="invalid_metric"
            )

if __name__ == '__main__':
    unittest.main()
