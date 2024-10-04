#!/bin/bash
#SBATCH --time=2-0:00:00       # Maximum job runtime (2 days)
#SBATCH --cpus-per-task=1      # Number of processor cores
#SBATCH --nodes=1              # Number of nodes
#SBATCH --partition=gpu        # Partition name
#SBATCH --gres=gpu:1           # GPU resources
#SBATCH --mem=256G             # Maximum memory
#SBATCH -J "Alignment"         # Job name
#SBATCH --mail-user=arushi17@iastate.edu  # Email address for notifications
#SBATCH --mail-type=BEGIN,END,FAIL        # Notification types

# ----------------------------
# Alignment.sh for NeuroXCode
# ----------------------------

# Detect if running on the cluster (SLURM) or locally
if [ -n "$SLURM_JOB_ID" ]; then
    echo "Running on the cluster..."
    module purge
    module load git
    module load python/3.11.9-i2aasxp
else
    echo "Running locally..."
fi

# Check if the first argument is provided for PROJECTDIR
if [ -z "$1" ]; then
    echo "Error: No project directory supplied."
    echo "Usage: ./alignment.sh /path/to/LatentConceptAnalysis"
    exit 1  # Exit with an error code
else
    export PROJECTDIR=$1
    echo "Project directory is set to: $PROJECTDIR"
fi

# Navigate to the project directory
cd "$PROJECTDIR" || { echo "Failed to navigate to $PROJECTDIR"; exit 1; }

# Set project paths
export NEUROX_CODE_ROOT="$PROJECTDIR/NeuroX/NeuroXCode"

# Activate the virtual environment
# There are multiple virtual environments in NeuroX, this is the correct one for alignment(neurox-env)
source "$PROJECTDIR/NeuroX/neurox-env/bin/activate"

# Check if activation succeeded
if [ $? -eq 0 ]; then
    echo "Virtual environment activated successfully."
else
    echo "Failed to activate virtual environment."
    exit 1
fi

# Define directories
alignmentDir="$NEUROX_CODE_ROOT/src/evaluation/"
dataDir="$NEUROX_CODE_ROOT/src/data/alignment_data/"

echo "Alignment directory: $alignmentDir"
echo "Data directory: $dataDir"

# Navigate to the alignment directory
cd "$alignmentDir" || { echo "Failed to navigate to $alignmentDir"; exit 1; }

# Run the alignment.py script with desired thresholds and methods
python alignment.py \
    --sentence-file "$dataDir/java.in" \
    --label-file "$dataDir/java.label" \
    --cluster-file "$dataDir/clusters-500.txt" \
    --thresholds 90 95 100 \
    --methods M1 M2

# Check if alignment.py ran successfully
if [ $? -ne 0 ]; then
    echo "alignment.py encountered an error."
    deactivate
    exit 1
fi

echo "Alignment completed successfully."

# Deactivate the virtual environment
deactivate