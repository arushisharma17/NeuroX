import RAID.tokens as tok
from typing import Pattern, Union, List
import neurox.data.extraction.transformers_extractor as transformers_extractor
import neurox.data.loader as loader
import re
import numpy as np
from collections import defaultdict

from typing import List, Tuple
from tree_sitter import Language, Parser
import tree_sitter_java as tsjava
import graphviz


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
        Given a list of tokens, their activations, and a multi class filter, create multiclass labeled dataset.
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
        - A list of corresponding multiclass labels 
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
        - {output_prefix}_labels.txt: A text file containing the multiclass labels.
        - {output_prefix}_activations.npy: A numpy file containing the activations.
    """
    # Save the files
    words, labels, activations = _create_multiclass_data(source_code_file_path=source_code_file_path, model_desc=model_desc, class_filters=class_filters, balance_data=balance_data)
    with open(f"{output_prefix}_words.txt", "w") as f:
        f.write("\n".join(words))

    with open(f"{output_prefix}_labels.txt", "w") as f:
        f.write("\n".join(labels))

    np.save(f"{output_prefix}_activations.npy", np.array(activations))



def extract_tokens_with_bio(node) -> Tuple[List[str], List[str]]:
    """
    Extracts tokens from the AST with BIO labeling.

    Parameters
    ----------
    node : tree_sitter.Node
        The root node of the AST.

    Returns
    -------
    Tuple[List[str], List[str]]
        - tokens: A list of tokens extracted from the AST.
        - bio_labels: A list of BIO labels corresponding to each token.
    """
    tokens = []
    bio_labels = []

    def recurse_node(node, label_prefix=None):
        if node.is_named:
            token_text = node.text.decode('utf-8')
            if label_prefix:
                bio_labels.append(f"B-{label_prefix}")
                tokens.append(token_text)
            else:
                bio_labels.append("O")
                tokens.append(token_text)
        for child in node.children:
            if node.is_named:
                child_label = label_prefix or node.type
                if child.is_named and len(child.children) > 0:
                    recurse_node(child, child_label)
                else:
                    child_text = child.text.decode('utf-8')
                    if len(child_text.split()) > 1:
                        bio_labels.append(f"B-{child_label}")
                        tokens.append(child_text.split()[0])
                        for part in child_text.split()[1:]:
                            bio_labels.append(f"I-{child_label}")
                            tokens.append(part)
                    else:
                        bio_labels.append(f"B-{child_label}")
                        tokens.append(child_text)
            else:
                recurse_node(child)

    recurse_node(node)
    return tokens, bio_labels


def create_bio_labeled_dataset(source_code: bytes, model_desc: str) -> Tuple[List[str], List[str]]:
    """
    Create BIO labeled dataset from source code.

    Parameters
    ----------
    source_code : bytes
        Source code to be parsed for BIO labeling.
    model_desc : str
        Description of the model used for token representation.

    Returns
    -------
    Tuple[List[str], List[str]]
        - A list of tokens.
        - A list of BIO labels corresponding to the tokens.
    """
    JAVA_LANGUAGE = Language(tsjava.language())
    parser = Parser(JAVA_LANGUAGE)
    tree = parser.parse(source_code)
    root_node = tree.root_node
    tokens, bio_labels = extract_tokens_with_bio(root_node)
    
    # Simulated activations (in a real case, this would come from a model)
    activations = np.random.rand(len(tokens), 3).tolist()

    return tokens, bio_labels, activations


def save_bio_dataset(output_prefix: str, tokens: List[str], bio_labels: List[str], activations: List[List[float]]):
    """
    Save BIO labeled tokens, labels, and activations to files.

    Parameters
    ----------
    output_prefix : str
        Prefix for the output files.
    tokens : List[str]
        Tokens extracted from the source code.
    bio_labels : List[str]
        BIO labels corresponding to the tokens.
    activations : List[List[float]]
        Activation vectors corresponding to the tokens.
    """
    with open(f"{output_prefix}_tokens.txt", "w") as f:
        f.write("\n".join(tokens))

    with open(f"{output_prefix}_bio_labels.txt", "w") as f:
        f.write("\n".join(bio_labels))

    np.save(f"{output_prefix}_activations.npy", np.array(activations))


# Example usage
tokens, bio_labels, activations = create_bio_labeled_dataset(source_code, "bert-base-uncased")
save_bio_dataset("java_output", tokens, bio_labels, activations)


def extract_tokens_with_positional_bio_levels(node) -> Tuple[List[str], List[List[str]]]:
    """
    Extracts tokens from the AST and generates positional BIO labels for all levels.
    
    Parameters
    ----------
    node : tree_sitter.Node
        The root node of the AST.

    Returns
    -------
    Tuple[List[str], List[List[str]]]
        - tokens: A list of tokens extracted from the AST.
        - bio_labels_levels: A list of lists where each inner list contains the positional BIO labels for a specific iteration/level.
    """
    tokens = []
    bio_labels_levels = []

    def recurse(node, current_label=None, level=0):
        if len(bio_labels_levels) <= level:
            bio_labels_levels.append([])

        if node.is_named:
            token_text = node.text.decode('utf-8')
            token_list = re.findall(r'\w+|[^\s\w]', token_text)  # Splits into words and symbols

            for i, token in enumerate(token_list):
                if level == 0:
                    tokens.append(token)
                if i == 0:
                    bio_labels_levels[level].append(f"B-{current_label}" if current_label else "O")
                else:
                    bio_labels_levels[level].append(f"I-{current_label}" if current_label else "O")

        for child in node.children:
            if node.is_named:
                recurse(child, node.type, level + 1)
            else:
                recurse(child, current_label, level)

    recurse(node)

    # Ensure all lists have the same length by filling with "O" labels
    max_length = len(tokens)
    for labels in bio_labels_levels:
        while len(labels) < max_length:
            labels.append("O")

    return tokens, bio_labels_levels


def build_cfg(node, graph=None, parent_id=None) -> List[Tuple[str, str]]:
    """
    Builds a Control Flow Graph (CFG) and optionally visualizes it using Graphviz.

    Parameters
    ----------
    node : tree_sitter.Node
        The root node of the AST.
    graph : graphviz.Digraph, optional
        Graph object to visualize the control flow.
    parent_id : str, optional
        Parent node ID, used for connecting edges.

    Returns
    -------
    List[Tuple[str, str]]
        A list of edges representing the control flow.
    """
    edges = []
    node_id = str(id(node))

    if graph is not None:
        label = f"{node.type} [{node.start_point}-{node.end_point}]"
        graph.node(node_id, label)
        if parent_id:
            graph.edge(parent_id, node_id)

    if node.type in ["if_statement", "while_statement", "for_statement"]:
        for child in node.children:
            child_id = str(id(child))
            edges.append((node_id, child_id))
            edges.extend(build_cfg(child, graph, node_id))
    else:
        for child in node.children:
            edges.extend(build_cfg(child, graph, parent_id))

    return edges


def visualize_ast(node, graph=None, parent_id=None):
    """
    Visualizes the Abstract Syntax Tree (AST) using Graphviz.

    Parameters
    ----------
    node : tree_sitter.Node
        The root node of the AST.
    graph : graphviz.Digraph, optional
        The Graphviz Digraph object used to visualize the AST.
    parent_id : str, optional
        The ID of the parent node, used to create edges between nodes.
    """
    node_id = str(id(node))
    label = f"{node.type} [{node.start_point}-{node.end_point}]"
    graph.node(node_id, label)
    if parent_id:
        graph.edge(parent_id, node_id)
    for child in node.children:
        visualize_ast(child, graph, node_id)


# Extract tokens and positional BIO levels
tokens, bio_labels_levels = extract_tokens_with_positional_bio_levels(root_node)
print(f"Tokens: {tokens}")
print(f"BIO Labels at All Levels: {bio_labels_levels}")

# Create and visualize the Control Flow Graph (CFG)
cfg_graph = graphviz.Digraph(format="png")
cfg_edges = build_cfg(root_node, cfg_graph)
cfg_graph.render("java_cfg")

# Visualize the AST using Graphviz
ast_graph = graphviz.Digraph(format="png")
visualize_ast(root_node, ast_graph)
ast_graph.render("java_ast")