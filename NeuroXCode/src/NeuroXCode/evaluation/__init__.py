# NeuroXCode/src/evaluation/__init__.py

# Import main functions from alignment.py
from .alignment import (
    load_sentences_and_labels,
    load_clusters,
    create_label_map,
    create_label_map_2,
    assign_labels_to_clusters,
    assign_labels_to_clusters_2,
    analyze_clusters,
    main as run_alignment
)

# Need to add other evaluation metrics here

__all__ = [
    'load_sentences_and_labels',
    'load_clusters',
    'create_label_map',
    'create_label_map_2',
    'assign_labels_to_clusters',
    'assign_labels_to_clusters_2',
    'analyze_clusters',
    'run_alignment',
]

# Version information
__version__ = '0.1.0'