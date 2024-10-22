# Testing the Clustering Module

This document outlines the process for testing the Clustering functionality in NeuroXCode.

## Overview

The clustering module can be tested using an interactive script that allows users to specify clustering parameters and execute the clustering process without directly interacting with the command line interface.

## Test Script Location

The test script is located at:
`LatentConceptAnalysis/NeuroX/NeuroXCode_Test/src/clustering/tests/test_clustering.py`

## How to Run the Test

1. Navigate to the directory containing `test_clustering.py`.
2. Run the script using Python:
   ```
   python test_clustering.py
   ```

## Test Process

The test script does the following:

1. Sets a predefined project directory for testing purposes.
2. Prompts the user for input on:
   - Layer number
   - Number of clusters
   - Which clustering methods to use (Agglomerative, KMeans, Leaders)
   - Tau value (if Leaders clustering is selected)
3. Constructs a `neuroxcode run_clustering` command based on user input.
4. Executes the constructed command.
5. Reports the success or failure of the clustering process.

## Example Usage

When you run the test script, you'll see prompts like this:

Layer Number?
Number of Clusters?
Use Agglomerative Clustering? (yes/no) [yes]:
Use KMeans Clustering? (yes/no) [yes]:
Use Leaders Clustering? (yes/no) [yes]:
Tau value for Leaders Clustering? [0.5]:
Executing command:
neuroxcode run_clustering /path/to/project/dir 1 5 --agglomerative --kmeans --leaders -t 0.5
Clustering test completed successfully.


## Notes

- The project directory is pre-set in the script. If you need to test with a different directory, modify the `project_dir` variable in the `run_clustering_test()` function.
- The script uses default values for each prompt. Press Enter to use the default value, or input your desired value.
- The script constructs and executes the command that would typically be run directly in the terminal, allowing for easy testing of the clustering functionality.

## Troubleshooting

- If the test fails, check the error message printed by the script. Common issues might include:
  - Missing activation files in the project directory
  - Invalid clustering parameters
  - Environment or dependency issues

For more detailed information about the clustering process, refer to the main Clustering module documentation.