import json
import codecs
from collections import Counter
from tqdm import tqdm
import neurox.data.loader as data_loader



def load_files(file_list):
    """
    Loads content from multiple files and concatenates them into a single list.

    Args:
        file_list (str): Comma-separated list of file paths.

    Returns:
        list: A list containing the concatenated data from the input files.
    """
    all_data = []
    for file in file_list.split(','):
        with open(file) as f:
            all_data.extend(json.load(f))
    return all_data


def load_activations_and_tokens(activations_file, tokens_file, labels_file, num_neurons_per_layer=768):
    """
    Loads activations and token data.

    Args:
        activations_file (str): Path to the activations JSON file.
        tokens_file (str): Path to the tokens input file.
        labels_file (str): Path to the labels file.
        num_neurons_per_layer (int, optional): Number of neurons per layer in the model. Default is 768.

    Returns:
        tuple: A tuple containing activations and token data loaded from the files.
    """
    print("Loading activations...")
    activations, num_layers = data_loader.load_activations(f'{activations_file}', num_neurons_per_layer=num_neurons_per_layer)

    print("Loading tokens and labels...")
    tokens = data_loader.load_data(f'{tokens_file}', f'{labels_file}', activations, 1000)

    return activations, tokens


def prepare_dataset(activations, tokens, output_prefix):
    """
    Prepares the dataset by combining tokens and activations.

    Args:
        tokens (dict): Token data loaded from the input files.
        activations (list): Activation values from the neural model.
        output_prefix (str): Prefix for the output files to be saved.

    Returns:
        list: A list containing the final token dataset.
    """
    print("Preparing dataset...")
    sentences = []
    labels = []
    selected_tokens = Counter()
    token_dataset = []

    for line_idx, label_line in tqdm(enumerate(tokens['target'])):
        sentences.append(" ".join(tokens['source'][line_idx]))
        labels.append(" ".join(label_line))
        for label_idx, label in enumerate(label_line):
            token = tokens['source'][line_idx][label_idx]
            selected_tokens[token] += 1
            token_acts = activations[line_idx][label_idx, :].tolist()

            final_tok_rep = f'{token}|||{selected_tokens[token]}|||{len(sentences)-1}|||{label_idx}'
            token_dataset.append((final_tok_rep, token_acts))

    # Save dataset and other outputs
    print("Writing datasets...")
    save_data(sentences, f'{output_prefix}-sentences.json')
    save_data(labels, f'{output_prefix}-labels.json')
    save_data(token_dataset, f'{output_prefix}-dataset.json')

    print("Dataset saved successfully.")
    return token_dataset


def count_word_frequencies(dataset):
    """
    Counts word frequencies from the dataset.

    Args:
        dataset (list): List of tokenized data with words and activations.

    Returns:
        dict: A dictionary mapping words to their respective frequencies.
    """
    word_count = {}
    for entry in dataset:
        # Each entry is a tuple with (token representation, activation values)
        # So, we extract the token part from the entry
        token_rep = entry[0]  # This is the token|||index|||sentence_idx part
        word = token_rep.split('|||')[0]  # Extract the actual word/token

        # Increment the word count
        word_count[word] = word_count.get(word, 0) + 1

    print(f"Word types: {len(word_count)}, Word tokens: {sum(word_count.values())}")
    return word_count


def get_pieces(line):
    """
    Splits a line into its components: word, count, and sentence index.

    Args:
        line (str): A line from the dataset file containing token details.

    Returns:
        list: A list containing the word, count, and sentence index.
    """
    pieces = line.rsplit("|||", 3)
    return pieces


def filter_by_frequency(dataset, word_count, min_freq, max_freq, del_freq):
    """
    Filters the dataset based on word frequency thresholds.

    Args:
        dataset (list): The dataset containing token and activation information.
        word_count (dict): A dictionary containing word frequencies.
        min_freq (int): Minimum frequency threshold to retain words.
        max_freq (int): Maximum frequency threshold to retain words.
        del_freq (int): Frequency threshold above which words are excluded.

    Returns:
        list: A filtered dataset with words that meet the frequency conditions.
    """
    curr_count = {}
    filtered_data = []

    for entry in dataset:
        word, _, sentence_idx, label_idx = get_pieces(entry[0])

        if word_count.get(word, 0) > del_freq:  # Skip most frequent words
            continue
        if word_count.get(word, 0) < min_freq:  # Skip rare words
            continue
        if curr_count.get(word, 0) >= max_freq:  # Skip if exceeding max frequency
            continue

        curr_count[word] = curr_count.get(word, 0) + 1
        filtered_data.append(entry)

    return filtered_data


def save_data(data, output_file):
    """
    Saves the given data to a JSON file.

    Args:
        data (any): The data to be saved.
        output_file (str): The path where the data will be saved.

    Returns:
        None
    """
    with open(output_file, 'w') as fp:
        json.dump(data, fp, ensure_ascii=False)


def process_dataset(input_file, sentence_file, min_freq, max_freq, del_freq, output_file):
    """
    Processes the dataset by counting word frequencies and filtering based on frequency.

    Args:
        input_file (str): Path to the input dataset file.
        sentence_file (str): Path to the sentence file.
        min_freq (int): Minimum frequency threshold for word inclusion.
        max_freq (int): Maximum frequency threshold for word inclusion.
        del_freq (int): Maximum allowable frequency before exclusion.
        output_file (str): Output file prefix for saving processed data.

    Returns:
        None
    """
    # Step 1: Load the dataset and sentences
    dataset = load_files(input_file)
    sentences = load_files(sentence_file)

    # Step 2: Save the concatenated sentences
    save_data(sentences, f"{output_file}_sentences.json")

    # Step 3: Count word frequencies
    word_count = count_word_frequencies(dataset)

    # Step 4: Filter dataset based on frequency
    filtered_dataset = filter_by_frequency(dataset, word_count, min_freq, max_freq, del_freq)

    # Step 5: Save the filtered dataset
    save_data(filtered_dataset, f"{output_file}_dataset.json")
    
#---------------------------extract_data.py ----------------------

import json
import numpy as np
from tqdm import tqdm

def read_json_file(input_file):
    """Reads the JSON input file and extracts tokens and points."""
    print("Reading " + input_file)
    tokens = []
    points = []
    with open(input_file, 'r') as f:
        dataset = json.load(f)
        for entry in tqdm(dataset, desc="Processing entries"):
            tokens.append(entry[0])  # Assuming tokens are in entry[0]
            points.append(entry[1])  # Assuming points are in entry[1]
    return tokens, np.array(points)

def save_vocab_file(output_path, output_vocab_file, tokens):
    """Saves the vocabulary tokens to a file."""
    if output_vocab_file is None:
        output_vocab_file = output_path + "/processed-vocab.npy"
    print(f"Saving vocab file: {output_vocab_file}")
    np.save(output_vocab_file, tokens)

def save_point_file(output_path, output_point_file, points):
    """Saves the points to a file."""
    if output_point_file is None:
        output_point_file = output_path + "/processed-point.npy"
    print(f"Saving point file: {output_point_file}")
    np.save(output_point_file, points)

def process_files(input_file, output_path, output_vocab_file=None, output_point_file=None):
    """Main function to process input and save vocab and point files."""
    tokens, points = read_json_file(input_file)

    save_vocab_file(output_path, output_vocab_file, tokens)
    save_point_file(output_path, output_point_file, points)

    print("Written vocab file and point file")



# Example usage
if __name__ == "__main__":
    # Step 1: Load activations and tokens
    activations_file = '/content/NeuroX/examples/test_layer_activations-layer1.json'
    tokens_file = '/content/NeuroX/examples/test.in'
    labels_file = '/content/NeuroX/examples/test.label'
    output_prefix = '/content/NeuroX/examples/test'

    activations, tokens = load_activations_and_tokens(activations_file, tokens_file, labels_file)

    # Step 2: Prepare dataset and save
    dataset = prepare_dataset(activations, tokens, output_prefix)

    # Step 3: Process dataset with frequency filtering
    input_file = '/content/NeuroX/examples/test-dataset.json'
    sentence_file = '/content/NeuroX/examples/test-sentences.json'
    output_file = '/content/NeuroX/examples/output_activations'

    # Frequency thresholds
    min_freq = 5
    max_freq = 50
    del_freq = 500000

    # Run the process
    process_dataset(input_file, sentence_file, min_freq, max_freq, del_freq, output_file)
