"""
Purpose: This file is the main file to run the clustering pipeline, and it is used in the __main__.py file
Author: Akhilesh Nevatia
Status: This is the main file that is used to run the clustering pipeline, and it is used in the __main__.py file, showing in the pipeline.
"""

import argparse
import os
import sys
from .get_clusters import run_clustering

def check_activation_files(project_dir, layer):
    # layer not realy used in the new approach, but keeping it incase we revert to the old approach
    # base_path = os.path.join(project_dir, 'CodeConceptNet', 'clusters', f'java_test/test_layer{layer}', 'activations')
    base_path = project_dir
    point_file = os.path.join(base_path, 'processed-point.npy')
    vocab_file = os.path.join(base_path, 'processed-vocab.npy')
    
    if not os.path.exists(point_file) or not os.path.exists(vocab_file):
        print(f"Error: Activation files not found in {base_path}")
        print("Make sure extraction and processing have been performed before running clustering.")
        sys.exit(1)

def main(project_dir, layer, clusters, agglomerative=False, kmeans=False, leaders=False, tau=None):
    # Check if project directory exists
    if not os.path.isdir(project_dir):
        print(f"Error: Project directory {project_dir} does not exist.")
        sys.exit(1)

    # Check if activation files exist
    check_activation_files(project_dir, layer)

    # Run clustering
    print("Running clustering algorithms...")
    clustering_methods = []
    if agglomerative:
        clustering_methods.append('agglomerative')
    if kmeans:
        clustering_methods.append('kmeans')
    if leaders:
        clustering_methods.append('leaders')

    # Check if at least one clustering method is specified
    if not clustering_methods:
        print("Error: At least one clustering method must be specified.")
        sys.exit(1)

    try:
        run_clustering(project_dir, layer, clusters, clustering_methods, tau)
    except Exception as e:
        print(f"Error during clustering: {str(e)}")
        sys.exit(1)

    print("Clustering completed successfully!")

#section below is to primarily test the run_clustering.py, without using the __main__.py file implementation ( used when we do neuroxcode run_clustering in the terminal )
# Example Usage here: python run_clustering.py /path/to/project 6 50 --agglomerative --kmeans --leaders -t 0.5
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run clustering pipeline.')
    parser.add_argument('project_dir', type=str, help='Path to the project directory')
    parser.add_argument('layer', type=int, help='Layer number (e.g., 1, 6, 12)')
    parser.add_argument('clusters', type=int, help='Number of clusters to generate')
    parser.add_argument('--agglomerative', action='store_true', help='Run Agglomerative Clustering')
    parser.add_argument('--kmeans', action='store_true', help='Run KMeans Clustering')
    parser.add_argument('--leaders', action='store_true', help='Run Leaders Clustering')
    parser.add_argument('-t', '--tau', type=float, help='Specify tau value for Leaders Clustering')

    args = parser.parse_args()

    main(args.project_dir, args.layer, args.clusters, args.agglomerative, args.kmeans, args.leaders, args.tau)
