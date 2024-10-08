import unittest
import numpy as np
import svgwrite
from visualization import visualize_activations, TransformersVisualizer

class TestVisualizationFunctions(unittest.TestCase):

    def setUp(self):
        # Sample data for testing
        self.tokens = ["This", "is", "a", "test", "."]
        self.activations = [0.1, -0.2, 0.3, -0.4, 0.0]

    def test_visualize_activations(self):
        # Test visualize_activations function
        svg = visualize_activations(self.tokens, self.activations)
        self.assertIsInstance(svg, svgwrite.Drawing)

    def test_visualize_activations_invalid_text_direction(self):
        # Test invalid text direction
        with self.assertRaises(AssertionError):
            visualize_activations(self.tokens, self.activations, text_direction="invalid")

    def test_visualize_activations_mismatched_lengths(self):
        # Test mismatched lengths of tokens and activations
        activations_short = [0.1, -0.2, 0.3]
        with self.assertRaises(AssertionError):
            visualize_activations(self.tokens, activations_short)

    def test_visualize_activations_filter_fn(self):
        # Test with filter_fn
        svg = visualize_activations(self.tokens, self.activations, filter_fn="top_tokens")
        self.assertIsInstance(svg, svgwrite.Drawing)

    def test_transformers_visualizer_initialization(self):
        # Test TransformersVisualizer class initialization
        visualizer = TransformersVisualizer('bert-base-uncased')
        self.assertIsNotNone(visualizer.model)
        self.assertIsNotNone(visualizer.tokenizer)

    def test_transformers_visualizer_call(self):
        # Test calling the visualizer
        # Note: This test requires downloading a model; consider mocking for faster tests
        visualizer = TransformersVisualizer('bert-base-uncased')
        tokens = ["Hello", "world", "!"]
        layer = 0
        neuron = 10
        svg = visualizer(tokens, layer, neuron)
        self.assertIsInstance(svg, svgwrite.Drawing)

if __name__ == '__main__':
    unittest.main()
