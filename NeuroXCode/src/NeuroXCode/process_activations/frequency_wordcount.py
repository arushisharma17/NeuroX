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
    wordCount = {}
    
    for file in files:
        f = codecs.open(file, encoding="utf-8")
        print("Reading file: ", file)
        for line in f.readlines():
            words = line.strip().split(' ')
            for word in words:
                if word in wordCount:
                    wordCount[word] += 1
                else:
                    wordCount[word] = 1
    return wordCount

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

def main():
    """
    Main entry point for the script. Parses command-line arguments, reads input files, computes word count, 
    saves the results, and prints statistics about the word occurrences.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--input-file', type=str, required=True, 
                        help="Comma-separated list of input files to be processed.")
    parser.add_argument('--output-file', type=str, default="output_activations.json",
                        help="File where the word count will be saved as a JSON.")

    args = parser.parse_args()
    files = args.input_file.split(',')

    wordCount = word_count_from_files(files)
    
    save_word_count(wordCount, args.output_file)
    print_statistics(wordCount)

# Ensure the main function runs when executed directly
if __name__ == "__main__":
    main()

