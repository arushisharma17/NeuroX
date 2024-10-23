import os
import numpy as np

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

# Set the output directory
output_dir = 'LatentConceptAnalysis/NeuroX/NeuroXCode_Test/src/clustering/test_directory/CodeConceptNet/clusters/java_test/activations'

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Generate and save synthetic data
num_points, num_dims, vocab_size = 1000, 100, 1000
points, vocab = generate_synthetic_data(num_points=num_points, num_dims=num_dims, vocab_size=vocab_size)

# Save the synthetic data
point_file = os.path.join(output_dir, 'processed-point.npy')
vocab_file = os.path.join(output_dir, 'processed-vocab.npy')

save_synthetic_data(points, vocab, point_file, vocab_file)

print(f"Synthetic data generated and saved to {output_dir}")
print(f"Points shape: {points.shape}")
print(f"Vocab size: {len(vocab)}")
