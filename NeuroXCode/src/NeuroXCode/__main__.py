import argparse
from . import extract_activations, process_activations
from .clustering import run_clustering

def main():
    parser = argparse.ArgumentParser(description="NeuroXCode Command-Line Interface")
    subparsers = parser.add_subparsers(dest="command", help="Subcommands for NeuroXCode")

    # Subcommand for activation extraction
    parser_extract = subparsers.add_parser("extract_activations", help="Extract activations from models")
    parser_extract.add_argument("--model", type=str, required=True, help="Model description (e.g., bert-base-uncased)")
    parser_extract.add_argument("--input", type=str, required=True, help="Input file containing sentences")
    parser_extract.add_argument("--output", type=str, required=True, help="Output file to save activations")
    parser_extract.add_argument("--layers", type=int, nargs="+", help="List of layers to extract activations from")

    # Subcommand for run_clustering
    parser_run_clustering = subparsers.add_parser("run_clustering", help="Run clustering on extracted activations")
    parser_run_clustering.add_argument("project_dir", type=str, help="Path to the project directory")
    parser_run_clustering.add_argument("layer", type=int, help="Layer number (e.g., 1, 6, 12)")
    parser_run_clustering.add_argument("clusters", type=int, help="Number of clusters to generate")
    parser_run_clustering.add_argument("--agglomerative", action="store_true", help="Run Agglomerative Clustering")
    parser_run_clustering.add_argument("--kmeans", action="store_true", help="Run KMeans Clustering")
    parser_run_clustering.add_argument("--leaders", action="store_true", help="Run Leaders Clustering")
    parser_run_clustering.add_argument("-t", "--tau", type=float, help="Specify tau value for Leaders Clustering")

    args = parser.parse_args()

    # Extract activations command
    if args.command == "extract_activations":
        extract_activations(args.model, args.input, args.output, layers=args.layers)

    # Process activations command
    elif args.command == "process_activations":
        process_activations()
    
    # Clustering command
    elif args.command == "run_clustering":
        run_clustering(
            args.project_dir,
            args.layer,
            args.clusters,
            agglomerative=args.agglomerative,
            kmeans=args.kmeans,
            leaders=args.leaders,
            tau=args.tau
        )

if __name__ == "__main__":
    main()
