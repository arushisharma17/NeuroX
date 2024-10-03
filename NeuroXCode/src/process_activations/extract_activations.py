import os
import argparse
from neurox.data.extraction.transformers_extractor import extract_representations
from utils import load_config  # Importing the load_config function from utils.py

def create_activations_directory(output_dir, dataset_name, model_name):
    """
    Creates the required directory structure for storing activations.

    Args:
        output_dir (str): The base directory for the output.
        dataset_name (str): Name of the dataset (e.g., java, cuda).
        model_name (str): Name of the model (e.g., microsoft/codebert-base).

    Returns:
        str: Path for the activations directory.
    """
    activations_dir = os.path.join(output_dir, dataset_name, model_name.replace('/', '-'), 'Activations')
    os.makedirs(activations_dir, exist_ok=True)
    print(f"Created activations directory: {activations_dir}")
    return activations_dir


def extract_layerwise_activations(model_name, working_file, layer, layer_dir):
    """
    Extracts layer-wise activations using NeuroX's `extract_representations` function.

    Args:
        model_name (str): The model name (e.g., microsoft/codebert-base).
        working_file (str): The input sentence file.
        layer (int): The specific layer to extract activations from.
        layer_dir (str): The directory where activations for the layer will be saved.
    """
    # Use a simple file name inside the layerX directory to avoid redundancy
    activation_file = os.path.join(layer_dir, f'activations.json')  # Only one "layerX" in the name

    # Extract activations directly using NeuroX
    print(f"Extracting layer-wise activations for layer {layer}...")
    extract_representations(
        model_name,  # Model to use for extraction
        working_file,  # Input sentence file
        output_file=activation_file,  # Output file where activations will be stored
        decompose_layers=True,  # Decompose layers flag
        filter_layers=str(layer),  # Specify the layer to extract
        output_type="json"  # Save the activations as JSON
    )

    print(f"Activations saved to {activation_file}")
    return activation_file

