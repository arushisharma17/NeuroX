# Running the Alignment Script

This document provides instructions on how to run the `run_alignment.py` script, which is designed to assign labels to clusters based on word labels in sentences across multiple layers of a neural network model. The script utilizes two methods (M1 and M2) to explore different strategies for label assignment.

## Prerequisites

- Python 3.x
- Required Libraries:
  - argparse
  - os
  - sys
  - subprocess
  - glob
  - csv
  - json
  - time
  - numpy
  - collections (Counter)
  - typing

## Directory Structure

Ensure your project follows this directory structure:

```
LatentConceptAnalysis/
├── NeuroX/
│   ├── NeuroXCode/
│   │   └── src/
│   │       └── NeuroXCode/
│   │           ├── data/
│   │           │   └── alignment_data/
│   │           │       ├── java.in
│   │           │       ├── java.label
│   │           │       └── clusters-500.txt
│   │           └── evaluation/
│   │               ├── run_alignment.py
│   │               └── alignment.py
│   └── temp/
│       └── outputs/
│           └── test/
│               └── microsoft-codebert-base/
│                   └── Activations/
│                       ├── layer0/
│                       │   └── activations-layer0.json
│                       ├── layer1/
│                       │   └── activations-layer1.json
│                       └── layer2/
│                           └── activations-layer2.json
```


## Running the Script

1. Navigate to the `evaluation` directory:
   ```
   cd /path/to/LatentConceptAnalysis/NeuroX/NeuroXCode/src/NeuroXCode/evaluation
   ```

2. Run the `run_alignment.py` script:
   ```
   python run_alignment.py /path/to/LatentConceptAnalysis
   ```

## Script Functionality

The `run_alignment.py` script performs the following tasks:

1. Locates necessary directories and files within the project structure.
2. Verifies the existence of required input files (java.in, java.label, clusters-500.txt).
3. Finds the activation files for each layer in the model.
4. Executes the `alignment.py` script for each layer, passing the appropriate arguments.

The `alignment.py` script:

1. Loads sentences, labels, and clusters from input files.
2. Processes activation data for each layer.
3. Applies two different methods (M1 and M2) for assigning labels to clusters.
4. Analyzes the results using various metrics.
5. Generates and saves reports for each layer.

## Output

For each layer, the script generates two output files in the corresponding layer directory:

1. `assigned_labels_all_methods_thresholds.json`: Contains the assigned labels for all methods and thresholds.
2. `analysis_results.csv`: A CSV file with the analysis results, including:
   - Threshold
   - Clusters Labeled
   - Tag Coverage
   - Overall Alignment Score
   - Major Labels
   - All Labels
   - None Labels
   - Unique Tags Identified

## Parameters

The script uses the following parameters:

- Thresholds: 90%, 95%, 100%
- Methods: M1, M2

## Notes

- Ensure that your virtual environment is activated before running the script.
- The script will process all layers found in the Activations directory.
- Runtime information and any errors encountered during execution will be printed to the console.

This command will run the alignment script on the provided input files, using both M1 and M2 methods, with thresholds of 90%, 95%, and 100%. The results will be analyzed and output as described in the previous sections.