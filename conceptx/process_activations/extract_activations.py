# extract_activations.py

import os
import argparse
from neurox.data.extraction.transformers_extractor import extract_activations_from_model
from utils import load_config  # Importing the load_config function from utils.py

def create_activations_directory(output_dir, dataset_name, model_name, layer):
    """
    Creates the required directory structure for storing activations for a specific layer.

    Args:
        output_dir (str): The base directory for the output.
        dataset_name (str): Name of the dataset (e.g., java, cuda).
        model_name (str): Name of the model (e.g., microsoft/codebert-base).
        layer (int): The layer number.

    Returns:
        str: Path for the activations directory.
    """
    activations_dir = os.path.join(output_dir, dataset_name, model_name.replace('/', '-'), 'Activations', f'layer{layer}')
    os.makedirs(activations_dir, exist_ok=True)
    print(f"Created activations directory for layer {layer}: {activations_dir}")
    return activations_dir

def extract_layerwise_activations(model_name, working_file, layer, activations_dir):
    """
    Extracts layer-wise activations using NeuroX's `extract_activations_from_model` function.

    Args:
        model_name (str): The model name (e.g., microsoft/codebert-base).
        working_file (str): The input sentence file.
        layer (int): The specific layer to extract activations from.
        activations_dir (str): The directory where activations will be saved.
    """
    # Create a subdirectory for each layer inside the activations directory
    layer_dir = os.path.join(activations_dir, f'layer{layer}')
    os.makedirs(layer_dir, exist_ok=True)
    print(f"Created directory for layer {layer}: {layer_dir}")
    
    # Define the activation file path inside the layer-specific subdirectory
    activation_file = os.path.join(layer_dir, f'activations-layer{layer}.json')
    
    # Extract activations directly using NeuroX
    print(f"Extracting layer-wise activations for layer {layer}...")
    extract_activations_from_model(
        model_name,  # Model to use for extraction
        working_file,  # Input sentence file
        output_file=activation_file,  # Output file where activations will be stored
        decompose_layers=True,  # Decompose layers flag
        filter_layers=[layer],  # Specify the layer to extract
        output_type="json"  # Save the activations as JSON
    )
    
    print(f"Activations saved to {activation_file}")
    return activation_file

def main(config):
    # Define paths and variables from config
    project_dir = config['paths']['project_dir']
    script_dir = config['paths']['script_dir']  # Path to scripts (not used in activations part right now)
    input_path = config['paths']['input_path']
    output_dir = config['paths']['output_dir']
    dataset_name = config['input']['dataset_name']
    model_name = config['model']['name'].replace('/', '-')
    input_file = config['input']['input_file']
    model = config['model']['name']
    layers = config['layers']

    # Working file location
    working_file = os.path.join(input_path, f"{input_file}.tok.sent_len")

    for layer in layers:
        # Create the directory structure for the activations
        activations_dir = create_activations_directory(output_dir, dataset_name, model_name, layer)

        # Copy input file to the working directory inside activations directory
        input_file_path = os.path.join(input_path, input_file)
        working_file_path = os.path.join(activations_dir, f"{input_file}.tok.sent_len")
        if not os.path.exists(working_file_path):
            print(f"Copying input file to {working_file_path}...")
            os.system(f"cp {input_file_path} {working_file_path}")

        # Extract layer-wise activations
        extract_layerwise_activations(model, working_file_path, layer, activations_dir)


def parse_args():
    """
    Parses command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to the JSON config file.')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    config = load_config(args.config)  # Loading config using the function from utils.py
    main(config)

