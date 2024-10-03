"""
Purpose: Refactoring getclusters.sh to a cleaner Python format in NeuroX
Author: Akhilesh Nevatia

Example usage:
Agglomerative Clustering:
python get_clusters.py /path/to/project 1 50 --agglomerative

KMeans Clustering:
python get_clusters.py /path/to/project 1 50 --kmeans

Leaders Clustering:
python get_clusters.py /path/to/project 1 50 --leaders -t 0.5

Multiple methods:
python get_clusters.py /path/to/project 1 50 --agglomerative --kmeans --leaders -t 0.5
"""

import argparse
import os
import sys
from ...algorithms.clustering.agglomerative import AgglomerativeClusteringPipeline
from ...algorithms.clustering.kmeans import KMeansClusteringPipeline
from ...algorithms.clustering.leaders import LeadersClusteringPipeline
import numpy as np

def run_clustering(project_dir, layer, clusters, clustering_methods, tau=None):
    # Set up environment variables
    os.environ['PROJECTDIR'] = project_dir
    os.environ['CODECONCEPTNET_ROOT'] = os.path.join(project_dir, 'CodeConceptNet')

    # Create directory name for the specific layer
    dir_name = f"java_test/test_layer{layer}"
    
    # Change to the appropriate directory
    os.chdir(os.path.join(os.environ['CODECONCEPTNET_ROOT'], 'clusters', dir_name))

    print("Creating Clusters!..........................")
    print(f"Current working directory: {os.getcwd()}")

    # Load processed activation data
    points = np.load('activations/processed-point.npy')
    vocab = np.load('activations/processed-vocab.npy')

    # Run selected clustering methods
    if 'agglomerative' in clustering_methods:
        os.makedirs('agglomerative', exist_ok=True)
        print("Running Agglomerative Clustering...")
        # Initialize Agglomerative Clustering Pipeline with specified parameters
        agglomerative = AgglomerativeClusteringPipeline(output_path='./agglomerative', num_clusters=clusters)
        # Execute the clustering pipeline on the loaded data
        agglomerative.run_pipeline(points, vocab)

    if 'kmeans' in clustering_methods:
        os.makedirs('kmeans', exist_ok=True)
        print("Running KMeans Clustering...")
        # Initialize K-Means Clustering Pipeline with specified number of clusters
        kmeans = KMeansClusteringPipeline(num_clusters=clusters)
        # Execute the clustering pipeline on the loaded data
        kmeans.run_pipeline(points, vocab)

    if 'leaders' in clustering_methods:
        os.makedirs('leaders', exist_ok=True)
        print("Running Leaders Clustering...")
        # Initialize Leaders Clustering Pipeline with specified parameters
        # Note: tau is optional and will be estimated if not provided
        leaders = LeadersClusteringPipeline(K=clusters, tau=tau, is_fast=True)
        # Execute the clustering pipeline on the loaded data
        leaders.run_pipeline(points, vocab)

    print("DONE!..........................................")

def main():
    # Set up command-line argument parser
    parser = argparse.ArgumentParser(description='Run clustering algorithms on activation data.')
    parser.add_argument('project_dir', type=str, help='Path to the project directory')
    parser.add_argument('layer', type=int, help='Layer number (e.g., 1, 6, 12)')
    parser.add_argument('clusters', type=int, help='Number of clusters to generate')
    parser.add_argument('--agglomerative', action='store_true', help='Run Agglomerative Clustering')
    parser.add_argument('--kmeans', action='store_true', help='Run KMeans Clustering')
    parser.add_argument('--leaders', action='store_true', help='Run Leaders Clustering')
    parser.add_argument('-t', '--tau', type=float, help='Specify tau value for Leaders Clustering')

    # Parse command-line arguments
    args = parser.parse_args()

    # Check if the project directory exists
    if not os.path.isdir(args.project_dir):
        print(f"Error: Project directory {args.project_dir} does not exist.")
        sys.exit(1)

    # Determine which clustering methods to run based on command-line arguments
    clustering_methods = []
    if args.agglomerative:
        clustering_methods.append('agglomerative')
    if args.kmeans:
        clustering_methods.append('kmeans')
    if args.leaders:
        clustering_methods.append('leaders')

    # Ensure at least one clustering method is specified
    if not clustering_methods:
        print("Error: At least one clustering method must be specified.")
        parser.print_help()
        sys.exit(1)

    # Run the clustering pipeline with the specified parameters
    run_clustering(args.project_dir, args.layer, args.clusters, clustering_methods, args.tau)


if __name__ == "__main__":
    main()