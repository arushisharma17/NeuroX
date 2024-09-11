import json
import codecs

def load_files(file_list):
    """Load content from multiple files and concatenate."""
    all_data = []
    for file in file_list.split(','):
        with open(file) as f:
            all_data.extend(json.load(f))
    return all_data

def count_word_frequencies(input_file):
    """Counts the word frequencies from the input file."""
    word_count = {}
    files = input_file.split(',')

    for file in files:
        with codecs.open(file, encoding="utf-8") as f:
            print(f"Reading file: {file}")
            for line in f.readlines():
                words = line.strip().split(' ')
                for word in words:
                    word_count[word] = word_count.get(word, 0) + 1
    
    print(f"Word types: {len(word_count)}, Word tokens: {sum(word_count.values())}")
    return word_count

def get_pieces(line):
    """Extract word, count, and sentence index from the input line."""
    pieces = line.rsplit("|||", 3)
    return pieces

def filter_by_frequency(dataset, word_count, min_freq, max_freq, del_freq):
    """Filter the dataset based on word frequencies."""
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
    """Save data to a JSON file."""
    with open(output_file, 'w') as fp:
        json.dump(data, fp, ensure_ascii=False)

def process_dataset(input_file, sentence_file, min_freq, max_freq, del_freq, output_file):
    """Main function to process the dataset by frequency filtering."""
    
    # Step 1: Count word frequencies
    word_count = count_word_frequencies(input_file)
    
    # Step 2: Load sentences and dataset
    sentences = load_files(sentence_file)
    dataset = load_files(input_file)

    # Step 3: Save the concatenated sentences
    save_data(sentences, f"{output_file}_sentences.json")
    
    # Step 4: Filter dataset based on frequency
    filtered_dataset = filter_by_frequency(dataset, word_count, min_freq, max_freq, del_freq)

    # Step 5: Save the filtered dataset
    save_data(filtered_dataset, f"{output_file}_dataset.json")
