import json
import numpy as np
from tqdm import tqdm
from collections import Counter
import NeuroXCode.process_activations.loader as data_loader
import os


class DatasetProcessor:
    @staticmethod
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

    @staticmethod
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

    def load_activations_and_tokens(self, activations_file, tokens_file, labels_file, num_neurons_per_layer=768):
        """
        Loads activations and token data.

        Args:
            activations_file (str): Full path to the activations JSON file.
            tokens_file (str): Full path to the tokens input file.
            labels_file (str): Full path to the labels file.
            num_neurons_per_layer (int, optional): Number of neurons per layer in the model. Default is 768.

        Returns:
            tuple: A tuple containing activations and token data loaded from the files.
        """
        print(f"Loading activations from: {activations_file}")
        activations, num_layers = data_loader.load_activations(activations_file, num_neurons_per_layer=num_neurons_per_layer)

        print(f"Loading tokens and labels from: {tokens_file} and {labels_file}")
        tokens = data_loader.load_data(tokens_file, labels_file, activations, 1000)

        return activations, tokens

    def prepare_dataset(self, activations, tokens, output_prefix):
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

        # Save dataset and other outputs with the generated prefix
        print("Writing datasets...")
        self.save_data(sentences, f'{output_prefix}_token_sentences.json')
        self.save_data(labels, f'{output_prefix}_token_labels.json')
        self.save_data(token_dataset, f'{output_prefix}_token_activations.json')

        print("Dataset saved successfully.")
        return token_dataset

    @staticmethod
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
