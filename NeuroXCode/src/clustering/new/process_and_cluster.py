import argparse
import os
import sys
from get_clusters import run_clustering

def check_activation_files(project_dir, layer):
    base_path = os.path.join(project_dir, 'CodeConceptNet', 'clusters', f'java_test/test_layer{layer}', 'activations')
    point_file = os.path.join(base_path, 'processed-point.npy')
    vocab_file = os.path.join(base_path, 'processed-vocab.npy')
    
    if not os.path.exists(point_file) or not os.path.exists(vocab_file):
        print(f"Error: Activation files not found in {base_path}")
        print("Make sure extraction has been performed before running clustering.")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description='Run clustering algorithms on extracted activations.')
    parser.add_argument('project_dir', type=str, help='Path to the project directory')
    parser.add_argument('layer', type=int, help='Layer number (e.g., 1, 6, 12)')
    parser.add_argument('clusters', type=int, help='Number of clusters to generate')
    parser.add_argument('--agglomerative', action='store_true', help='Run Agglomerative Clustering')
    parser.add_argument('--kmeans', action='store_true', help='Run KMeans Clustering')
    parser.add_argument('--leaders', action='store_true', help='Run Leaders Clustering')
    parser.add_argument('-t', '--tau', type=float, help='Specify tau value for Leaders Clustering')

    args = parser.parse_args()

    # Check if project directory exists
    if not os.path.isdir(args.project_dir):
        print(f"Error: Project directory {args.project_dir} does not exist.")
        sys.exit(1)

    # Check if activation files exist
    check_activation_files(args.project_dir, args.layer)

    # Run clustering
    print("Running clustering algorithms...")
    clustering_methods = []
    if args.agglomerative:
        clustering_methods.append('agglomerative')
    if args.kmeans:
        clustering_methods.append('kmeans')
    if args.leaders:
        clustering_methods.append('leaders')

    # Check if at least one clustering method is specified
    if not clustering_methods:
        print("Error: At least one clustering method must be specified.")
        parser.print_help()
        sys.exit(1)

    try:
        run_clustering(args.project_dir, args.layer, args.clusters, clustering_methods, args.tau)
    except Exception as e:
        print(f"Error during clustering: {str(e)}")
        sys.exit(1)

    print("Clustering completed successfully!")

if __name__ == "__main__":
    main()