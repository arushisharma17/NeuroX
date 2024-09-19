### ConceptX directory structure

```
/your/project_directory/
└── NeuroX/
    └── conceptx/
        ├── __init__.py
        ├── process_activations/               # Directory for processing activations
        │   ├── activations_processor.py       # Script for processing activations
        │   ├── extract_activations.py         # Script for extracting activations from models
        │   ├── frequency_filtering.py         # Script for filtering data based on word frequencies
        │   ├── frequency_wordcount.py         # Script for calculating word frequencies
        │   ├── utils.py                       # Utility functions used across scripts
        │   └── __init__.py                    # Marks the directory as a Python module
        ├── algorithms/                        # Directory for algorithms
        │   ├── clustering/                    # Clustering algorithms
        │   │   ├── __init__.py
        │   └── __init__.py
        ├── utilities/                         # Utility functions or scripts
        └── evaluation/                        # Directory for evaluation scripts
            ├── __init__.py                    # Marks the evaluation directory as a module
            ├── metrics.py                     # Script for handling evaluation metrics
            └── alignment.py                   # Script for alignment evaluations
```
