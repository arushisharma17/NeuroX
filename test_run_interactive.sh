#!/bin/bash

module purge
module load micromamba
module load git

# Check if the first argument is provided for PROJECTDIR
if [ -z "$1" ]; then
    echo "Error: No project directory supplied."
    echo "Usage: ./setup.sh /path/to/LatentConceptAnalysis"
    exit 1  # Exit with an error code
else
    export PROJECTDIR=$1
    echo "Project directory is set to: $PROJECTDIR"
fi

cd $PROJECTDIR

# Setup micromamba
if [ ! -d "micromamba" ]; then
    echo "micromamba does not exist"
fi

export MAMBA_ROOT_PREFIX=$PROJECTDIR/micromamba

export NEUROX_DIR=$PROJECTDIR/NeuroX

cd $NEUROX_DIR

eval "$(micromamba shell hook --shell=bash)"
micromamba activate env_activations

# Check if 'env_activations' environment works correctly
if micromamba activate env_activations; then
    echo "Environment 'env_activations' activated successfully."
#    python -m pip install git+https://github.com/huggingface/transformers
#    python -m pip install -e .
    
    python conceptx/process_activations/extract_activations.py --config temp/config.json 

    # List installed packages in the environment
    echo "Deactivating environment 'env_activations':"
    #micromamba list
    micromamba deactivate
else
    echo "Failed to activate environment 'env_activations'."
    exit 1
fi
