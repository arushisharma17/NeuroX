# NeuroXCode

A toolkit for neuron analysis in deep NLP models.

## Features
- Activation extraction
- Clustering algorithms
- Process activations

## Installation

To install, use the following command:
```
pip install neuroxcode
```

## Packaging the Project

To generate a pip-installable tarball and wheel file:

1. Run the following command to create the distribution files:
   ```
   python3 setup.py sdist bdist_wheel
   ```

2. You will find the `.tar.gz` and `.whl` files in the `dist/` directory.

## Installing the Package

To install the package from the generated wheel file, use:
```
pip install dist/neuroxcode-0.0.1-py3-none-any.whl
```

## Usage

Once installed, you can use the command-line interface. For example, to extract activations:
```
neuroxcode extract_activations --model bert-base-uncased --input input.txt --output output.txt
```

This will run the activation extraction from the specified model and save the results to the output file.
