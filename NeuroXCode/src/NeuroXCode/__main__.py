import argparse
from . import extract_activations, process_activations
from .src.algorithms.clustering import AgglomerativeClustering, KMeansClustering, LeadersClustering

def main():
    parser = argparse.ArgumentParser(description="NeuroXCode Command-Line Interface")
    subparsers = parser.add_subparsers(dest="command", help="Subcommands for NeuroXCode")

    # Subcommand for activation extraction
    parser_extract = subparsers.add_parser("extract_activations", help="Extract activations from models")
    parser_extract.add_argument("--model", type=str, required=True, help="Model description (e.g., bert-base-uncased)")
    parser_extract.add_argument("--input", type=str, required=True, help="Input file containing sentences")
    parser_extract.add_argument("--output", type=str, required=True, help="Output file to save activations")
    parser_extract.add_argument("--layers", type=int, nargs="+", help="List of layers to extract activations from")

    # Subcommand for processing activations
    parser_process = subparsers.add_parser("process_activations", help="Process activations")
    # Additional options for process_activations could be added here if needed

    # Subcommand for clustering algorithms
    parser_clustering = subparsers.add_parser("clustering", help="Run clustering algorithms")
    parser_clustering.add_argument("--method", type=str, required=True, choices=["agglomerative", "kmeans", "leaders"],
                                   help="Clustering method")
    parser_clustering.add_argument("--input", type=str, required=True, help="Input data for clustering")
    parser_clustering.add_argument("--output", type=str, required=True, help="Output file for clustering results")

    args = parser.parse_args()

    # Extract activations command
    if args.command == "extract_activations":
        extract_activations(args.model, args.input, args.output, layers=args.layers)

    # Process activations command
    elif args.command == "process_activations":
        process_activations()

    # Clustering command
    elif args.command == "clustering":
        if args.method == "agglomerative":
            AgglomerativeClustering(args.input, args.output)
        elif args.method == "kmeans":
            KMeansClustering(args.input, args.output)
        elif args.method == "leaders":
            LeadersClustering(args.input, args.output)

if __name__ == "__main__":
    main()
