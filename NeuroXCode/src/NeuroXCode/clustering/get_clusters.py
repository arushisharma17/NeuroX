"""
Purpose: Refactoring getclusters.sh to a cleaner Python format in NeuroX
Author: Akhilesh Nevatia
Usage: This class is just a clean way of representing the get_clusters.sh file in Python, and to use this we have the run_clustering.py file or the process_and_cluster.py file
Status: Can be thought of as an abstraction which gets used in the run_clustering.py file or the process_and_cluster.py file, which are the main files shown in the pipeline.
"""

import os
import numpy as np
from NeuroXCode.algorithms.clustering.agglomerative import AgglomerativeClusteringPipeline
from NeuroXCode.algorithms.clustering.kmeans import KMeansClusteringPipeline
from NeuroXCode.algorithms.clustering.leaders import LeadersClusteringPipeline
'''
Going down the Approach where user directly points us to the directory with the the required files ( processed-point.npy and processed-vocab.npy )
'''
def run_clustering(project_dir, layer, clusters, clustering_methods, tau=None):
    # os.environ['PROJECTDIR'] = project_dir
    # os.environ['CODECONCEPTNET_ROOT'] = os.path.join(project_dir, 'CodeConceptNet')

    # # Create directory name for the specific layer
    # dir_name = f"java_test/test_layer{layer}"
    
    # # Change to the appropriate directory
    # os.chdir(os.path.join(os.environ['CODECONCEPTNET_ROOT'], 'clusters', dir_name))

    # # Create the full path for the layer-specific directory
    # layer_dir = os.path.join(project_dir, 'activations', dir_name)

    # 
    os.chdir(project_dir)

    print("Creating Clusters!..........................")
    print(f"Current working directory: {os.getcwd()}")

    # Load processed activation data
    points = np.load('processed-point.npy')
    vocab = np.load('processed-vocab.npy')

    # Run selected clustering methods
    if 'agglomerative' in clustering_methods:
        # os.makedirs('agglomerative', exist_ok=True)
        agglomerative_dir = os.path.join(project_dir, 'agglomerative')
        os.makedirs(agglomerative_dir, exist_ok=True)
        print("Running Agglomerative Clustering...")
        # Initialize Agglomerative Clustering Pipeline with specified parameters
        agglomerative = AgglomerativeClusteringPipeline(output_path=agglomerative_dir, num_clusters=clusters)
        # Execute the clustering pipeline on the loaded data
        agglomerative.run_pipeline(points, vocab)

    if 'kmeans' in clustering_methods:
        # os.makedirs('kmeans', exist_ok=True)
        kmeans_dir = os.path.join(project_dir, 'kmeans')
        os.makedirs(kmeans_dir, exist_ok=True)
        print("Running KMeans Clustering...")
        # Initialize K-Means Clustering Pipeline with specified number of clusters
        kmeans = KMeansClusteringPipeline(output_path=kmeans_dir, num_clusters=clusters)
        # Execute the clustering pipeline on the loaded data
        kmeans.run_pipeline(points, vocab)

    if 'leaders' in clustering_methods:
        # os.makedirs('leaders', exist_ok=True)
        leaders_dir = os.path.join(project_dir, 'leaders')
        os.makedirs(leaders_dir, exist_ok=True)
        print("Running Leaders Clustering...")
        # Initialize Leaders Clustering Pipeline with specified parameters
        # Note: tau is optional and will be estimated if not provided
        leaders = LeadersClusteringPipeline(output_path=leaders_dir, K=clusters, tau=tau, is_fast=True)
        # Execute the clustering pipeline on the loaded data
        leaders.run_pipeline(points, vocab)

    print("DONE!..........................................")

