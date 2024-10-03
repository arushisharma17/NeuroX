#frequency_filtering.py

import argparse
import json


def get_pieces(line):
    """
    Splits a string line into word, word count, sentence count, and label using the '|||' separator.
    Args:
        line (str): The string line to split.
    Returns:
        list: A list of the four components split from the line.
    """
    pieces = []
    end_idx = len(line)
    for _ in range(3):
        sep_idx = line[:end_idx].rfind("|||")
        pieces.append(line[sep_idx + 3:end_idx])
        end_idx = sep_idx
    pieces.append(line[:end_idx])
    return list(reversed(pieces))


def load_word_count(frequency_file):
    """
    Loads the word frequency from a JSON file.
    Args:
        frequency_file (str): The path to the frequency file.
    Returns:
        dict: A dictionary of word counts.
    """
    with open(frequency_file) as f:
        print("Loading frequency")
        word_count = json.load(f)
    print(f"Len of word dict: {len(word_count)}")
    return word_count


def load_sentences(sentence_file):
    """
    Loads the sentences from one or more JSON files.
    Args:
        sentence_file (str): A comma-separated list of sentence files.
    Returns:
        list: A list of sentences from the files.
        dict: A dictionary with file sizes.
    """
    files = sentence_file.split(',')
    sentences = []
    files_size = {}
    for idx, file in enumerate(files):
        with open(file) as f:
            data = json.load(f)
            filesize = len(data)
            sentences.extend(line.strip() for line in data)
        files_size[idx] = filesize
    print("File sizes:", files_size)
    return sentences, files_size


def filter_dataset(dataset_files, word_count, files_size, min_freq, max_freq, del_freq, print_info=True):
    """
    Filters the dataset based on word frequencies.
    Args:
        dataset_files (str): A comma-separated list of dataset files.
        word_count (dict): A dictionary of word counts.
        files_size (dict): A dictionary with the sizes of the files.
        min_freq (int): Minimum frequency threshold for inclusion.
        max_freq (int): Maximum frequency threshold for inclusion.
        del_freq (int): Delete frequency threshold for exclusion.
    Returns:
        list: The filtered dataset.
        tuple: Stats about maxskip, minskip, delskip, maxskips, minskips, delskips.
    """
    dataset_wordcount = {}
    curr_count = {}
    output = []
    maxskip, minskip, delskip = 0, 0, 0
    maxskips, minskips, delskips = set(), set(), set()

    dataset_sentencecount = 0
    for idx, file in enumerate(dataset_files.split(',')):
        print("Loading", file)
        with open(file) as f:
            dataset = json.load(f)
            print(len(dataset), len(word_count))
            for entry in dataset:
                word, d_wordcount, d_sentencecount, label_idx = get_pieces(entry[0])
                d_sentencecount = int(d_sentencecount)
                dataset_wordcount[word] = dataset_wordcount.get(word, 0) + 1
                d_sentencecount += dataset_sentencecount
                entry = (f"{word}|||{dataset_wordcount[word]}|||{d_sentencecount}|||{label_idx}", entry[1])
                if word in word_count and word_count[word] > del_freq:
                    if print_info: print(f"Delete word {word}")
                    delskip += 1
                    delskips.add(word)
                    continue
                if word in word_count and word_count[word] >= min_freq:
                    if curr_count.get(word, 0) < max_freq:
                        output.append(entry)
                        curr_count[word] = curr_count.get(word, 0) + 1
                    else:
                        if print_info: print(f"Crossed max frequency: {entry[0]}")
                        maxskip += 1
                        maxskips.add(word)
                else:
                    if print_info: print(f"Skipping word with low frequency: {entry[0]}")
                    minskip += 1
                    minskips.add(word)

        dataset_sentencecount += files_size[idx]

    return output, (maxskip, minskip, delskip, maxskips, minskips, delskips)


def save_to_file(data, output_file):
    """
    Saves data to a JSON file.
    Args:
        data: The data to save.
        output_file (str): The output file path.
    """
    with open(output_file, 'w') as fp:
        json.dump(data, fp, ensure_ascii=False)
    print(f"Saved data to {output_file}")


def print_statistics(word_count, maxskip, minskip, delskip, maxskips, minskips, delskips):
    """
    Prints statistics about the filtered dataset.
    Args:
        word_count (dict): The original word count dictionary.
        maxskip (int): Number of tokens skipped based on maximum frequency.
        minskip (int): Number of tokens skipped based on minimum frequency.
        delskip (int): Number of tokens skipped based on delete frequency.
        maxskips (set): Words skipped based on maximum frequency.
        minskips (set): Words skipped based on minimum frequency.
        delskips (set): Words skipped based on delete frequency.
    """
    print("Limit Max types:", maxskips)
    print("Skipped Min types:", minskips)
    print("Skipped frequent types:", delskips)
    print(f"Total word types before dropping: {len(word_count)}")
    print(f"Total word tokens before dropping: {sum(word_count.values())}")
    print(f"Tokens skipped based on Max freq: {maxskip}")
    print(f"Tokens skipped based on Min freq: {minskip}")
    print(f"Tokens skipped based on Del freq: {delskip}")
    print(f"Types skipped based on Max freq: {len(maxskips)}")
    print(f"Types skipped based on Min freq: {len(minskips)}")
    print(f"Types skipped based on Del freq: {len(delskips)}")
    print(f"Remaining Tokens: {sum(word_count.values()) - maxskip - minskip - delskip}")
    print(f"Remaining Types: {len(word_count) - len(minskips) - len(delskips)}")
