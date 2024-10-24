#frequency_wordcount.py

import codecs
import argparse
import json


def word_count_from_files(files):
    """
    Reads multiple files and computes the word count for each word.

    Args:
        files (list): List of file paths to be read.

    Returns:
        dict: A dictionary where the keys are words and the values are the count of occurrences.
    """
    all_data = []
    for file in files:
        with open(file) as f:
            all_data.extend(json.load(f))


    word_count = {}
    for entry in all_data:
        token_rep = entry[0]  # This is the token|||index|||sentence_idx part
        word = token_rep.split('|||')[0]  # Extract the actual word/token
        word_count[word] = word_count.get(word, 0) + 1

    # print(f"Word types: {len(word_count)}, Word tokens: {sum(word_count.values())}")
    return word_count


def save_word_count(wordCount, output_file):
    """
    Saves the word count dictionary to a JSON file.

    Args:
        wordCount (dict): A dictionary where the keys are words and the values are the count of occurrences.
        output_file (str): The path to the output file where the word count will be saved.
    """
    print("Saving output file")
    with open(output_file, 'w') as fp:
        json.dump(wordCount, fp)


def print_statistics(wordCount):
    """
    Prints various statistics about the word counts, including the number of types (unique words),
    total tokens (words), and the number of types occurring less than 2, 3, 4, and 5 times.

    Args:
        wordCount (dict): A dictionary where the keys are words and the values are the count of occurrences.
    """
    print("######### Singletons #############")
    print([k for k,v in wordCount.items() if v < 2])
    print("#####################################")
    
    print("Types in vocab: ", len(wordCount))
    print("Tokens in vocab: ", sum(wordCount.values()))

    lessthan5 = 0
    lessthan4 = 0
    lessthan3 = 0
    lessthan2 = 0

    for k,v in wordCount.items():
        if v < 2:
            lessthan5 += 1
            lessthan4 += 1
            lessthan3 += 1
            lessthan2 += 1
        elif v < 3:
            lessthan5 += 1
            lessthan4 += 1
            lessthan3 += 1
        elif v < 4:
            lessthan5 += 1
            lessthan4 += 1
        elif v < 5:
            lessthan5 += 1
    
    print("Types less than 2: ", lessthan2)
    print("Types less than 3: ", lessthan3)
    print("Types less than 4: ", lessthan4)
    print("Types less than 5: ", lessthan5)

