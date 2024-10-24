import os
import argparse
from NeuroXCode.data.extraction.transformers_extractor import extract_representations


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


def extract_all_layer_activations(model_name, input_dir, output_dir, dataset):
    """
    Extracts all layer-wise activations using NeuroX's `extract_representations` function.

    Args:
        model_name (str): The model name (e.g., microsoft/codebert-base).
        input_dir (str): input directory to be filled by dataset choice.
        output_dir (int, tuple): output directory where activations will be placed.
        dataset (str): dataset name.
    """
    working_file = os.path.join(input_dir, f'{dataset}/{dataset}.in')
    # Use a simple file name inside the layerX directory to avoid redundancy
    activations_dir = os.path.join(output_dir, dataset, model_name.replace('/', '-'), 'Activations')
    activation_file = os.path.join(activations_dir, f'activations.json')  # Only one "layerX" in the name

    # Extract activations directly using NeuroX
    # print(f"Extracting layer-wise activations for layer {layer}...")
    extract_representations(
        model_name,  # Model to use for extraction
        working_file,  # Input sentence file
        output_file=activation_file,  # Output file where activations will be stored
        decompose_layers=True,  # Decompose layers flag
        filter_layers=str(),  # Specify the layer to extract
        output_type="json"  # Save the activations as JSON
    )

    for file in os.listdir(activations_dir):
        layer = ((file.split('.')[0]).split('-'))[-1]
        os.makedirs(os.path.join(activations_dir, layer), exist_ok=True)
        if file.find('.json') > -1:
            os.rename(os.path.join(activations_dir, file), os.path.join(activations_dir, layer, file))
        # print('Files:', file)

    print(f"Activations saved to {activation_file}")
    return activation_file


def extract_layerwise_activations(model_name, working_file, layer, layer_dir):
    """
    Extracts layer-wise activations using NeuroX's `extract_representations` function.

    Args:
        model_name (str): The model name (e.g., microsoft/codebert-base).
        working_file (str): The input sentence file.
        layer (int, tuple): The specific layer to extract activations from. Leave blank to extract all layers.
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

