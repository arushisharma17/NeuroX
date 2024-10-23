import argparse
import os
from NeuroXCode.process_activations.extract_activations import main as extract_activations
from ..process_and_cluster import main as process_and_cluster

def main():
    parser = argparse.ArgumentParser(description="Run extraction and clustering pipeline")
    parser.add_argument("project_dir", type=str, help="Path to the project directory")
    parser.add_argument("config_file", type=str, help="Path to the config file")
    parser.add_argument("layer", type=int, help="Layer number")
    parser.add_argument("clusters", type=int, help="Number of clusters")
    parser.add_argument("clustering_methods", nargs="+", help="Clustering methods to use")
    args = parser.parse_args()

    # Run extraction
    extract_activations(args.project_dir)

    # Run clustering
    process_and_cluster(args.project_dir, args.layer, args.clusters, args.clustering_methods)

if __name__ == "__main__":
    main()