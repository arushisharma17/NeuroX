# Activation Extraction and Clustering Workflow

This document outlines the workflow for extracting neural network activations and performing clustering on them.

## Workflow Overview

run_extraction_and_clustering.sh

│

├─ test_run_interactive.sh

│  └─ extract_activations.py

│

├─ process_and_cluster.py

└─ get_clusters.py


## Script Descriptions

### 1. run_extraction_and_clustering.sh

- **Purpose**: Main entry point that orchestrates the entire process.
- **Functions**:
  - Sets up the environment
  - Calls `test_run_interactive.sh` for activation extraction
  - Calls `process_and_cluster.py` for clustering

### 2. test_run_interactive.sh

- **Purpose**: Prepares the environment and extracts activations.
- **Functions**:
  - Loads necessary modules
  - Sets up the project environment
  - Calls `extract_activations.py`

### 3. extract_activations.py

- **Purpose**: Extracts neural network activations.
- **Output**: Saves extracted activations to disk (typically as .npy files).

### 4. process_and_cluster.py

- **Purpose**: Manages the clustering process.
- **Functions**:
  - Verifies the existence of extracted activation files
  - Parses command-line arguments for clustering parameters
  - Calls functions from `get_clusters.py`

### 5. get_clusters.py

- **Purpose**: Performs actual clustering operations.
- **Functions**:
  - Loads extracted activations from disk
  - Implements various clustering algorithms (Agglomerative, KMeans, Leaders)
  - Saves clustering results

## Key Features

1. **Single Command Execution**: Extract activations and perform clustering in one step.
2. **Separation of Concerns**: Clear distinction between extraction and clustering processes.
3. **Code Reusability**: Utilizes existing `get_clusters.py` for clustering operations.
4. **Error Handling**: Checks for necessary files before proceeding with clustering.
5. **Flexibility**: Supports multiple clustering methods with customizable parameters.

## Usage

chmod +x run_extraction_and_clustering.sh // Build the executable 
./run_extraction_and_clustering.sh <project_dir> <config_file> <layer> <clusters> <clustering_methods...> // now run it 

Example:

chmod +x run_extraction_and_clustering.sh
./run_extraction_and_clustering.sh /path/to/project /path/to/config.json 1 50 --agglomerative --kmeans


This command extracts activations and runs both Agglomerative and KMeans clustering on layer 1, creating 50 clusters.

## Notes for Future Reference

- Ensure all scripts are in their correct locations within the project structure.
- The extraction process saves files in a specific directory structure. Verify this structure if encountering file not found errors.
- Clustering methods and parameters can be easily extended in `get_clusters.py`.
- For large datasets, consider implementing checkpointing in the extraction process.

## Troubleshooting

- If clustering fails, check if activation files were correctly generated in the expected location.
- Verify that the correct environment is activated before running the main script.
- For memory issues, consider reducing the number of clusters or implementing batch processing.
