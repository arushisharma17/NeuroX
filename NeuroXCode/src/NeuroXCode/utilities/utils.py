import os
import dill as pickle
import numpy as np
import logging
import platform
import sys
import json
import csv
from datetime import datetime
import sklearn

# Save clustering results (reusable for all clustering methods)
def save_clustering_results(clustering, clusters, output_path, K, ref = ''):
    """
    Saves the results of the clustering including the model and the cluster assignments.

    Args:
        clustering (object): Fitted clustering model (any clustering method).
        clusters (dict): Cluster assignments with words mapped to clusters.
        output_path (str): Path to save the results.
        K (int): Number of clusters.
        ref (str): Reference or suffix for file names.
    """
    logging.info(f"Saving clustering results for {K} clusters...")
    model_file = os.path.join(output_path, f"model-{K}-clustering{ref}.pkl")
    with open(model_file, "wb") as fp:
        pickle.dump(clustering, fp)

    cluster_output = "\n".join([f"{word}|||{key}" for key, words in clusters.items() for word in words])
    cluster_file = os.path.join(output_path, f"clusters-{K}{ref}.txt")
    with open(cluster_file, 'w') as f:
        f.write(cluster_output)

# Load data (can be reused for any clustering method)
def load_data(point_file=None, vocab_file=None, num_points=100, num_dims=5, vocab_size=100, output_path='./output'):
    """
    Loads data either from provided files or generates synthetic data.

    Args:
        point_file (Optional[str]): Path to file containing preloaded points data.
        vocab_file (Optional[str]): Path to file containing preloaded vocab data.
        num_points (int): Number of synthetic data points (if generating synthetic data).
        num_dims (int): Number of dimensions for each data point (if generating synthetic data).
        vocab_size (int): Size of the vocabulary (if generating synthetic data).
        output_path (str): Directory to store generated synthetic data files.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Points and vocabulary arrays.
    """
    try:
        if point_file and vocab_file:
            logging.info("Loading data from provided files.")
            points = np.load(point_file)
            vocab = np.load(vocab_file)
        else:
            logging.info("Generating synthetic data.")
            points, vocab = generate_synthetic_data(num_points=num_points, num_dims=num_dims, vocab_size=vocab_size)
            save_synthetic_data(points, vocab,
                                point_file=os.path.join(output_path, 'synthetic_points.npy'),
                                vocab_file=os.path.join(output_path, 'synthetic_vocab.npy'))
        return points, vocab
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        return None, None

    # Generate synthetic data (reusable for any clustering method)
def generate_synthetic_data(num_points=100, num_dims=5, vocab_size=100):
    """
    Generates synthetic data points and vocabulary for clustering.

    Args:
        num_points (int): Number of data points to generate.
        num_dims (int): Dimensionality of each data point.
        vocab_size (int): Vocabulary size.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Generated data points and corresponding vocab.
    """
    points = np.random.rand(num_points, num_dims)
    vocab = np.array([f'word_{i}' for i in range(vocab_size)])
    return points, vocab

# Save synthetic data (reusable for any clustering method)
def save_synthetic_data(points, vocab, point_file, vocab_file):
    """
    Saves generated synthetic data to specified file paths.

    Args:
        points (np.ndarray): Synthetic data points.
        vocab (np.ndarray): Synthetic vocabulary.
        point_file (str): Path to save points data.
        vocab_file (str): Path to save vocab data.
    """
    np.save(point_file, points)
    np.save(vocab_file, vocab)

# Logging functions using the logging module
def log_environment_info():
    """Log environment and system details using logging."""
    env_info = (
        f"Python Version: {sys.version}\n"
        f"NumPy Version: {np.__version__}\n"
        f"SciKit-Learn Version: {sklearn.__version__}\n"
        f"System: {platform.system()} {platform.release()}\n"
        f"Processor: {platform.processor()}\n"
        f"Machine: {platform.machine()}\n"
    )
    logging.info(env_info)

def log_input_data(points, vocab):
    """Log summary of input data."""
    input_summary = f"Points Shape: {points.shape}\nVocab Size: {len(vocab)}\n"
    logging.info(input_summary)

def log_clustering_params(K):
    """Log clustering parameters."""
    clustering_params = f"Number of Clusters: {K}\nDistance Metric: Euclidean\nLinkage: Ward\n"
    logging.info(clustering_params)

def log_cluster_summary(clusters):
    """Log a summary of the clustering results."""
    cluster_summary = "Cluster Summary:\n"
    for cluster_id, members in clusters.items():
        cluster_summary += f"Cluster {cluster_id}: {len(members)} items\n"
    logging.info(cluster_summary)

def log_runtime(start_time, end_time):
    """Log the runtime of clustering."""
    runtime = f"Clustering Runtime: {end_time - start_time:.2f} seconds\n"
    logging.info(runtime)

def log_start_time():
    """Log the start time."""
    start_time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    logging.info(f"Start Time: {start_time_str}")

def log_end_time():
    """Log the end time."""
    end_time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    logging.info(f"End Time: {end_time_str}")

def log_clustering_process(points, vocab, K, start_time, end_time, clusters):
    """Combined logging function for the entire clustering process."""
    log_start_time()
    log_environment_info()
    log_input_data(points, vocab)
    log_clustering_params(K)
    log_runtime(start_time, end_time)
    log_cluster_summary(clusters)
    log_end_time()

# Example main function to demonstrate usage
def main():
    # Set up logging to output to a file
    output_path = './output'
    os.makedirs(output_path, exist_ok=True)
    logging.basicConfig(filename=os.path.join(output_path, 'clustering.log'),
                        level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

    # Load or generate synthetic data
    points, vocab = load_data(num_points=10, num_dims=3, vocab_size=10, output_path=output_path)

    # Example clusters and model (using synthetic data)
    clustering_model = {"dummy_model": "example"}  # Placeholder for a clustering model
    clusters = {i: [vocab[i]] for i in range(len(vocab))}  # Example clusters for testing

    # Perform the logging of the entire process
    start_time = datetime.now()
    log_clustering_process(points, vocab, K=5, start_time=start_time, end_time=datetime.now(), clusters=clusters)

    # Save clustering results
    save_clustering_results(clustering_model, clusters, output_path, K=5, ref='_test')

    logging.info("Test completed.")

if __name__ == "__main__":
    main()

