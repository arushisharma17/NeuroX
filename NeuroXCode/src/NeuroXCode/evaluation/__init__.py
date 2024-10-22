# NeuroXCode/src/evaluation/__init__.py

# Import main functions from alignment.py
from .alignment import (
    load_sentences_and_labels,
    load_clusters,
    load_activations,
    process_activations,
    filter_label_map,
    create_label_map,
    create_label_map_2,
    extract_words_items,
    assign_labels_to_clusters,
    assign_labels_to_clusters_2,
    group_clusters,
    analyze_clusters,
    generate_final_report,
    main as run_alignment
)

__all__ = [
    'load_sentences_and_labels',
    'load_clusters',
    'load_activations',
    'process_activations',
    'filter_label_map',
    'create_label_map',
    'create_label_map_2',
    'extract_words_items',
    'assign_labels_to_clusters',
    'assign_labels_to_clusters_2',
    'group_clusters',
    'analyze_clusters',
    'generate_final_report',
    'run_alignment',
]

# Version information
__version__ = '0.1.0'