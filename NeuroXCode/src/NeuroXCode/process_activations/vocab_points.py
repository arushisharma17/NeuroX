from tqdm import tqdm
import numpy as np

def generate_vocab_and_points(filtered_output, output_prefix):
    """
    Generates the vocab and points files from the token dataset.

    Args:
        token_dataset (list): A list containing the tokens and their activations.
        output_prefix (str): Prefix for the output files to be saved.

    Returns:
        None
    """
    print("Generating vocab and points...")

    # Extract vocab and points from token dataset
    vocab = []
    points = []

    for entry in tqdm(filtered_output, desc="Processing token activations"):
        token_rep, activations = entry
        token = token_rep.split('|||')[0]  # Extract the actual token (without metadata)
        vocab.append(token)
        points.append(activations)

    # Convert points to a numpy array for saving
    points = np.array(points)

    # Save vocab and points to files
    vocab_file = f"{output_prefix}_processed_vocab.npy"
    points_file = f"{output_prefix}_processed_points.npy"

    print(f"Saving vocab to {vocab_file}")
    np.save(vocab_file, vocab)

    print(f"Saving points to {points_file}")
    np.save(points_file, points)
    print("Vocab and points files generated successfully.")