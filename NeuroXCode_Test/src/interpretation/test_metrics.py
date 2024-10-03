import unittest
import numpy as np
from metrics import (
    accuracy,
    f1,
    accuracy_and_f1,
    pearson,
    spearman,
    pearson_and_spearman,
    matthews_corrcoef,
    compute_score
)

class TestMetrics(unittest.TestCase):

    def setUp(self):
        # Create sample predictions and labels
        self.preds_classification = np.array([0, 1, 1, 0, 1, 0])
        self.labels_classification = np.array([0, 1, 0, 0, 1, 1])

        self.preds_regression = np.array([2.5, 0.0, 2.1, 1.8])
        self.labels_regression = np.array([3.0, -0.5, 2.0, 2.0])

    def test_accuracy(self):
        acc = accuracy(self.preds_classification, self.labels_classification)
        self.assertTrue(0.0 <= acc <= 1.0)

    def test_f1(self):
        f1_score = f1(self.preds_classification, self.labels_classification)
        self.assertTrue(0.0 <= f1_score <= 1.0)

    def test_accuracy_and_f1(self):
        acc_f1 = accuracy_and_f1(self.preds_classification, self.labels_classification)
        self.assertTrue(0.0 <= acc_f1 <= 1.0)

    def test_pearson(self):
        pearson_score = pearson(self.preds_regression, self.labels_regression)
        self.assertTrue(-1.0 <= pearson_score <= 1.0)

    def test_spearman(self):
        spearman_score = spearman(self.preds_regression, self.labels_regression)
        self.assertTrue(-1.0 <= spearman_score <= 1.0)

    def test_pearson_and_spearman(self):
        combined_score = pearson_and_spearman(self.preds_regression, self.labels_regression)
        self.assertTrue(-1.0 <= combined_score <= 1.0)

    def test_matthews_corrcoef(self):
        mcc_score = matthews_corrcoef(self.preds_classification, self.labels_classification)
        self.assertTrue(-1.0 <= mcc_score <= 1.0)

    def test_compute_score(self):
        acc = compute_score(self.preds_classification, self.labels_classification, 'accuracy')
        self.assertTrue(0.0 <= acc <= 1.0)
        with self.assertRaises(ValueError):
            compute_score(self.preds_classification, self.labels_classification, 'invalid_metric')

if __name__ == '__main__':
    unittest.main()
