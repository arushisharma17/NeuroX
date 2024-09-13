from typing import List
def gather_tokens(file_path: str) -> List[List[str]]:
    """
    Reads source code from a file and splits it into tokens.

    Parameters
    ----------
    file_path : str
        The path to the source code file.

    Returns
    -------
    List[List[str]]
        A list of lists, where each inner list contains tokens from a line of code.
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()
    tokens = [line.strip().split() for line in lines]
    return tokens
