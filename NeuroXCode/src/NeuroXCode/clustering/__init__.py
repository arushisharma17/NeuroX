"""Clustering module for NeuroXCode.

This module idelly provides functionality for extracting activations,
processing data, and performing clustering on neural network representations.
"""

from NeuroXCode.process_activations.extract_activations import main as extract_activations
from .process_and_cluster import main as process_and_cluster
from .get_clusters import run_clustering

__all__ = ['extract_activations', 'process_and_cluster', 'run_clustering']

# Version of the clustering module
__version__ = "0.1.0"