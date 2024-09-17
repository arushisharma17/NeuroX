import RAID.tokens as tok
from typing import Pattern, Union, List
import neurox.data.extraction.transformers_extractor as transformers_extractor
import neurox.data.loader as loader
import re
import numpy as np
from collections import defaultdict
def createBinaryClassification(source_code_file_path:str,model_desc:str,binary_filter:Union[set,Pattern,callable])-> tuple[List[str], List[str], List[List[float]]]:
    """
        Given a list of tokens, their activations, and a binary_filter, create a binary labeled dataset.
        A binary filter can be a set, regex, or a function.
        Parameters
        --------------
        source_code_file_path : str
            File path to source code which will be used for the binary classification.

        binary_filter : set or Pattern or callable
            A set of words, a regex object, or a function that defines the positive class.

        model_desc : str
            Model description; can either be a model name like ``bert-base-uncased``,
            a comma separated list indicating <model>,<tokenizer> (since 1.0.8),
            or a path to a trained model
        
        Returns
        ------------
        Tuple[List[str], List[str], List[List[float]]]
        A tuple containing:
        - A list of words (tokens)
        - A list of corresponding binary labels ("positive" or "negative")
        - A list of corresponding activation vectors
        
    """
    binary_filter = _checkfilter(binary_filter)
    tokens = tok.gather_tokens(source_code_file_path)
    transformers_extractor.extract_representations(model_desc=model_desc, input_corpus=source_code_file_path,output_file="output.json")
    activations, _ = loader.load_activations("output.json")
    if isinstance(binary_filter, set):
        filter_fn = lambda x: x in binary_filter
    elif isinstance(binary_filter, Pattern):
        filter_fn = lambda x: binary_filter.match(x)
    elif callable(binary_filter):  # Updated to check if it's callable
        filter_fn = binary_filter
    else:
        raise NotImplementedError("ERROR: The binary_filter must be a set, a regex pattern, or a callable function.")

    positive_class_words = []
    positive_class_activations = []
    negative_class_words = []
    negative_class_activations = []

    print("Creating binary dataset ...")
    for s_idx, sentences in enumerate(tokens):
        for w_idx, word in enumerate(sentences):
            if filter_fn(word):
                positive_class_words.append(word)
                positive_class_activations.append(activations[s_idx][w_idx])
            else:
                negative_class_words.append(word)
                negative_class_activations.append(activations[s_idx][w_idx])

    if len(negative_class_words) == 0 or len(positive_class_words) == 0:
        raise ValueError("ERROR: Positive or Negative class examples are zero")
    elif len(negative_class_words) < len(positive_class_words):
        print("WARNING: The negative class examples are less than the positive class examples")
        print("Positive class examples: ", len(positive_class_words), "Negative class examples: ", len(negative_class_words))


    print("Number of Positive examples: ", len(positive_class_words))

    labels = ['positive'] * len(positive_class_words) + ['negative'] * len(negative_class_words)
    return (positive_class_words + negative_class_words, labels, positive_class_activations + negative_class_activations)

def _checkfilter(binary_filter:Union[set,Pattern,callable]):
    # Handle binary filter input (e.g., converting string input to regex or set)
    if callable(binary_filter):
        # Use the callable filter as is
        pass
    elif isinstance(binary_filter, str):
        if binary_filter.startswith("re:"):
            binary_filter = re.compile(binary_filter[3:])
        elif binary_filter.startswith("set:"):
            binary_filter = set(binary_filter[4:].split(","))
        else:
            raise NotImplementedError("String filter must start with 're:' for regex or 'set:' for a set of words.")
    else:
        raise TypeError("Filter must be a callable, or start with 're:' for regex, or 'set:' for a set of words.")
    return binary_filter


def createBinaryClassification(source_code_file_path:str,output_prefix:str,model_desc:str,binary_filter:Union[set,Pattern,callable]):
    """
    Overloaded method with an additional parameter "output_prefix" which specifies the path of the output files.
    Parameters
    -----------
    source_code_file_path : str
        File path to source code which will be used for the binary classification.

    binary_filter : set or Pattern or callable
        A set of words, a regex object, or a function that defines the positive class.

    model_desc : str
        Model description; can either be a model name like ``bert-base-uncased``,
        a comma separated list indicating <model>,<tokenizer> (since 1.0.8),
        or a path to a trained model

    output_prefix : str
        Prefix for the output files that will be saved as the result of this script.
    Returns
    -------
    None
        The function saves the following files:
        - {output_prefix}_words.txt: A text file containing the words (tokens).
        - {output_prefix}_labels.txt: A text file containing the binary labels.
        - {output_prefix}_activations.npy: A numpy file containing the activations.
    """
    
    words, labels, activations = createBinaryClassification(source_code_file_path=source_code_file_path, model_desc=model_desc, binary_filter=binary_filter)
    with open(f"{output_prefix}_words.txt", "w") as f:
        f.write("\n".join(words))

    with open(f"{output_prefix}_labels.txt", "w") as f:
        f.write("\n".join(labels))

    np.save(f"{output_prefix}_activations.npy", np.array(activations))
    
    
def _create_multiclass_data(source_code_file_path: str, model_desc: str, class_filters: dict, balance_data = False) ->tuple[List[str], List[str], List[List[float]]]:
    """
        Given a list of tokens, their activations, and a binary_filter, create multiclass labeled dataset.
        A multi class filter is a dict with sets, functions or patterns as keys which map to classes
        Parameters
        --------------
        source_code_file_path : str
            File path to source code which will be used for the binary classification.

        `class_filters` : Union[set,Pattern,callable]
        is a d where keys are class names and values are filters that define
        which tokens belong to each class. The filter can be:
        1. A set of specific token texts (e.g., {"if", "while"}).
        2. A regular expression (Pattern) to match token texts.
        3. A callable function that takes a token's type and text, returning True if the token matches.
        This allows flexible classification of tokens into multiple classes based on their type or content.

        model_desc : str
            Model description; can either be a model name like ``bert-base-uncased``,
            a comma separated list indicating <model>,<tokenizer> (since 1.0.8),
            or a path to a trained model
        
        Returns
        ------------
        Tuple[List[str], List[str], List[List[float]]]
        A tuple containing:
        - A list of words (tokens)
        - A list of corresponding binary labels ("positive" or "negative")
        - A list of corresponding activation vectors
        
    """
    class_words = defaultdict(list)
    class_activations = defaultdict(list)
    tokens = tok.gather_tokens(source_code_file_path)
    transformers_extractor.extract_representations(model_desc=model_desc, input_corpus=source_code_file_path,output_file="output.json")
    activations, _ = loader.load_activations("output.json")
    def get_class(token_type, token_text):
        for class_name, filter_fn in class_filters.items():
            if isinstance(filter_fn, set) and token_text in filter_fn:
                return class_name
            elif isinstance(filter_fn, Pattern) and filter_fn.match(token_text):
                return class_name
            elif callable(filter_fn) and filter_fn(token_type, token_text):
                return class_name
        return "negative"

    print("Creating multi-class dataset ...")
    for w_idx, (token_type, token_text) in enumerate(tokens):
        class_name = get_class(token_type, token_text)
        class_words[class_name].append(token_text)
        class_activations[class_name].append(activations[0][w_idx])

    for class_name in class_filters.keys():
        if len(class_words[class_name]) == 0:
            raise ValueError(f"ERROR: No examples found for class '{class_name}'")

    if balance_data:
        min_examples = min(len(words) for words in class_words.values())
        for class_name in class_filters.keys():
            class_words[class_name] = class_words[class_name][:min_examples]
            class_activations[class_name] = class_activations[class_name][:min_examples]

    words = []
    labels = []
    activations_list = []

    for class_name in class_filters.keys():
        words.extend(class_words[class_name])
        labels.extend([class_name] * len(class_words[class_name]))
        activations_list.extend(class_activations[class_name])

    return words, labels, activations_list

def _create_multiclass_data(source_code_file_path: str, model_desc: str, class_filters: dict, output_prefix:str, balance_data = False):
    """
    Overloaded method with an additional parameter "output_prefix" which specifies the path of the output files.
    Parameters
    -----------
    source_code_file_path : str
        File path to source code which will be used for the binary classification.

    `class_filters` : Union[set,Pattern,callable]
    is a d where keys are class names and values are filters that define
    which tokens belong to each class. The filter can be:
    1. A set of specific token texts (e.g., {"if", "while"}).
    2. A regular expression (Pattern) to match token texts.
    3. A callable function that takes a token's type and text, returning True if the token matches.
    This allows flexible classification of tokens into multiple classes based on their type or content.


    model_desc : str
        Model description; can either be a model name like ``bert-base-uncased``,
        a comma separated list indicating <model>,<tokenizer> (since 1.0.8),
        or a path to a trained model
        
    output_prefix : str
        Prefix for the output files that will be saved as the result of this script.
    Returns
    -------
    None
        The function saves the following files:
        - {output_prefix}_words.txt: A text file containing the words (tokens).
        - {output_prefix}_labels.txt: A text file containing the binary labels.
        - {output_prefix}_activations.npy: A numpy file containing the activations.
    """
    # Save the files
    words, labels, activations = _create_multiclass_data(source_code_file_path=source_code_file_path, model_desc=model_desc, class_filters=class_filters, balance_data=balance_data)
    with open(f"{output_prefix}_words.txt", "w") as f:
        f.write("\n".join(words))

    with open(f"{output_prefix}_labels.txt", "w") as f:
        f.write("\n".join(labels))

    np.save(f"{output_prefix}_activations.npy", np.array(activations))

