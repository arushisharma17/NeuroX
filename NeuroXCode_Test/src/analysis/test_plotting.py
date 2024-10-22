import unittest
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend suitable for testing
import matplotlib.pyplot as plt
import numpy as np
from plotting import plot_accuracies_per_tag, plot_distributedness, plot_accuracies

class TestPlottingFunctions(unittest.TestCase):

    def setUp(self):
        # Sample data for testing
        self.title = "Test Plot"
        self.accuracies = {
            'Experiment 1': {'__OVERALL__': 0.9, 'Class1': 0.85, 'Class2': 0.95},
            'Experiment 2': {'__OVERALL__': 0.88, 'Class1': 0.8, 'Class2': 0.92}
        }
        self.top_neurons_per_tag = {
            'Class1': [1, 2, 3],
            'Class2': [4, 5],
            'Class3': [6]
        }
        self.overall_acc = 0.9
        self.top_10_acc = 0.85
        self.random_10_acc = 0.75
        self.bottom_10_acc = 0.65
        self.top_15_acc = 0.86
        self.random_15_acc = 0.76
        self.bottom_15_acc = 0.66
        self.top_20_acc = 0.87
        self.random_20_acc = 0.77
        self.bottom_20_acc = 0.67

    def test_plot_accuracies_per_tag(self):
        # Test plotting accuracies per tag
        fig = plot_accuracies_per_tag(self.title, **self.accuracies)
        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_distributedness(self):
        # Test plotting distributedness
        fig = plt.figure()
        plot_distributedness(self.title, self.top_neurons_per_tag)
        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_accuracies(self):
        # Test plotting accuracies
        fig = plt.figure()
        plot_accuracies(
            self.title,
            self.overall_acc,
            self.top_10_acc,
            self.random_10_acc,
            self.bottom_10_acc,
            self.top_15_acc,
            self.random_15_acc,
            self.bottom_15_acc,
            self.top_20_acc,
            self.random_20_acc,
            self.bottom_20_acc
        )
        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_accuracies_per_tag_no_data(self):
        # Test plotting with no data
        with self.assertRaises(TypeError):
            plot_accuracies_per_tag(self.title)

if __name__ == '__main__':
    unittest.main()
