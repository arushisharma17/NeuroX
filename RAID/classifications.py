import RAID.tokens as tok
from typing import Pattern, Union, List
import neurox.data.extraction.transformers_extractor as transformers_extractor
import neurox.data.loader as loader
import re
import numpy as np
def createBinaryClassification(source_code_file_path:str,model_desc:str,binary_filter:Union[set,Pattern,callable])-> (List[str], List[str], List[List[float]]):
    """
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
    Parameters
    -----------
    output_prefix : str
        Prefix for the output files that will be saved as the result of this script.
    
    """
    words, labels, activations = createBinaryClassification(source_code_file_path=source_code_file_path,model_desc=model_desc, binary_filter=binary_filter)
    with open(f"{output_prefix}_words.txt", "w") as f:
        f.write("\n".join(words))

    with open(f"{output_prefix}_labels.txt", "w") as f:
        f.write("\n".join(labels))

    np.save(f"{output_prefix}_activations.npy", np.array(activations))
    
    


    
