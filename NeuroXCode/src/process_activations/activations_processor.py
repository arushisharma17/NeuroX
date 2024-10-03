import json
import numpy as np
from tqdm import tqdm
from collections import Counter
import neurox.data.loader as data_loader
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

    def generate_vocab_and_points(self, token_dataset, output_prefix):
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

        for entry in tqdm(token_dataset, desc="Processing token activations"):
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

    def count_word_frequencies(self, dataset):
        """
        Counts word frequencies from the dataset.

        Args:
            dataset (list): List of tokenized data with words and activations.

        Returns:
            dict: A dictionary mapping words to their respective frequencies.
        """
        word_count = {}
        for entry in dataset:
            token_rep = entry[0]  # This is the token|||index|||sentence_idx part
            word = token_rep.split('|||')[0]  # Extract the actual word/token
            word_count[word] = word_count.get(word, 0) + 1

        print(f"Word types: {len(word_count)}, Word tokens: {sum(word_count.values())}")
        return word_count

    def filter_by_frequency(self, dataset, word_count, min_freq, max_freq, del_freq):
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
            word, _, sentence_idx, label_idx = self.get_pieces(entry[0])

            if word_count.get(word, 0) > del_freq:  # Skip most frequent words
                continue
            if word_count.get(word, 0) < min_freq:  # Skip rare words
                continue
            if curr_count.get(word, 0) >= max_freq:  # Skip if exceeding max frequency
                continue

            curr_count[word] = curr_count.get(word, 0) + 1
            filtered_data.append(entry)

        return filtered_data

    def process_dataset(self, input_file, sentence_file, min_freq, max_freq, del_freq, output_prefix):
        """
        Processes the dataset by counting word frequencies, filtering based on frequency, and generating vocab and points.

        Args:
            input_file (str): Full path to the input dataset file.
            sentence_file (str): Full path to the sentence file.
            min_freq (int): Minimum frequency threshold for word inclusion.
            max_freq (int): Maximum frequency threshold for word inclusion.
            del_freq (int): Maximum allowable frequency before exclusion.
            output_prefix (str): Prefix for the output files to be saved.

        Returns:
            None
        """
        # Step 1: Load the dataset and sentences
        dataset = self.load_files(input_file)
        sentences = self.load_files(sentence_file)

        # Step 2: Save the concatenated sentences
        self.save_data(sentences, f"{output_prefix}_sentences.json")

        # Step 3: Count word frequencies
        word_count = self.count_word_frequencies(dataset)

        # Step 4: Filter dataset based on frequency
        filtered_dataset = self.filter_by_frequency(dataset, word_count, min_freq, max_freq, del_freq)

        # Step 5: Save the filtered dataset
        self.save_data(filtered_dataset, f"{output_prefix}_filtered_dataset.json")

        # print(filtered_dataset)
        # Step 6: Generate vocab and points files from the filtered dataset
        self.generate_vocab_and_points(filtered_dataset, output_prefix)

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
