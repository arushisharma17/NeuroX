#!/bin/bash
#SBATCH --time=4-0:00:00     # Maximum job runtime of 4 days
#SBATCH --cpus-per-task=1    # Number of processor cores per task
#SBATCH --nodes=4            # Number of nodes requested (4 nodes)
#SBATCH -J "Setup"           # Job name
#SBATCH --mail-user=arushi17@iastate.edu  # Email address for job notifications
#SBATCH --mail-type=BEGIN    # Send email at job start
#SBATCH --mail-type=END      # Send email at job end
#SBATCH --mail-type=FAIL     # Send email on job failure

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
    mkdir micromamba
fi
export MAMBA_ROOT_PREFIX=$PROJECTDIR/micromamba

export NEUROX_DIR=$PROJECTDIR/NeuroX

cd $NEUROX_DIR

eval "$(micromamba shell hook --shell=bash)"

# Optionally remove existing environment if needed (commented by default)
# micromamba remove -n env_activations --all

# Create environment to extract representations using NeuroX
micromamba env create -f $NEUROX_DIR/env_activations.yml
echo "env_activations created"

# Check if 'env_activations' environment works correctly
if micromamba activate env_activations; then
    echo "Environment 'env_activations' activated successfully."
    python -m pip install git+https://github.com/huggingface/transformers
    python -m pip install -e .

    # List installed packages in the environment
    echo "Listing installed packages in 'env_activations':"
    micromamba list
else
    echo "Failed to activate environment 'env_activations'."
    exit 1
fi
