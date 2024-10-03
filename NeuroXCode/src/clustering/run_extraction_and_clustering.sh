#!/bin/bash

# Check if all required arguments are provided
if [ "$#" -lt 5 ]; then
    echo "Usage: $0 <project_dir> <config_file> <layer> <clusters> <clustering_methods...>"
    echo "Example: $0 /path/to/project /path/to/config.json 1 50 --agglomerative --kmeans"
    exit 1
fi

PROJECT_DIR=$1
CONFIG_FILE=$2
LAYER=$3
CLUSTERS=$4
shift 4  # Shift past the first four arguments

# Run test_run_interactive.sh for environment setup and activation extraction
./temp/test_run_interactive.sh "$PROJECT_DIR"

# Activate the environment again (as test_run_interactive.sh deactivates it at the end)
eval "$(micromamba shell hook --shell=bash)"
micromamba activate env_activations

# Run process_and_cluster.py for clustering
python conceptx/process_activations/process_and_cluster.py "$PROJECT_DIR" "$LAYER" "$CLUSTERS" "$@"

# Deactivate the environment
micromamba deactivate