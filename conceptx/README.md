<img src="https://github.com/user-attachments/assets/e1a4db28-bb4f-4a44-a908-6437b555c754" alt="Overview" width="400"/>
<!-- Link to overview image: https://docs.google.com/drawings/d/1K24l9K9m-CJ6qdF7HHaJ0cjkJ8Xqpfx7PYOxdMe8mK0/edit?usp=sharing -->
<img src="https://github.com/user-attachments/assets/6a993b70-8163-4500-ab38-1b31324da6b6" alt="Second Image" width="400"/>




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
