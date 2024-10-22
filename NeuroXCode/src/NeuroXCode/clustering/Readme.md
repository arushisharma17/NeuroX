# Clustering Module

This module provides functionality for performing clustering on neural network representations.

## Files and Their Functions

### 1. run_clustering.py

- **Purpose**: Main entry point for the clustering pipeline.
- **Functions**:
  - Parses command-line arguments
  - Checks for the existence of activation files
  - Initiates the clustering process
- **Usage**: Called by the main NeuroXCode CLI

### 2. get_clusters.py

- **Purpose**: Implements the actual clustering algorithms.
- **Functions**:
  - Loads processed activation data
  - Runs selected clustering methods (Agglomerative, KMeans, Leaders)
  - Saves clustering results

### 3. __init__.py

- **Purpose**: Defines the public interface of the clustering module.
- **Functions**:
  - Exposes the `run_clustering` function for use in other parts of NeuroXCode

## Flow of Implementation

1. The user initiates the clustering process through the NeuroXCode CLI.
2. The `__main__.py` file in the root directory calls the `run_clustering` function from this module.
3. `run_clustering.py` checks for the necessary files and prepares the clustering parameters.
4. `get_clusters.py` is then called to perform the actual clustering operations.
5. Results are saved in the appropriate directories within the project structure.

## Usage

The clustering module is typically used through the NeuroXCode CLI:

neuroxcode run_clustering <project_dir> <layer> <clusters> [--agglomerative] [--kmeans] [--leaders] [-t TAU]

Example:

neuroxcode run_clustering /path/to/project 1 50 --agglomerative --kmeans

## Notes

- Ensure that activation extraction and processing have been performed before running clustering.
- The module supports multiple clustering methods that can be run simultaneously.
- Clustering results are saved in method-specific directories (e.g., 'agglomerative', 'kmeans', 'leaders').

## Troubleshooting

- If clustering fails, verify that the activation files exist in the expected location.
- For memory issues, consider reducing the number of clusters or implementing batch processing.
