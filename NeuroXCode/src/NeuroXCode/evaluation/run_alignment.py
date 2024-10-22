import os
import sys
import subprocess
import argparse
import glob

def find_directory(start_path, target_dir):
    for root, dirs, files in os.walk(start_path):
        if target_dir in dirs:
            return os.path.join(root, target_dir)
    return None

def run_alignment(project_dir):
    project_dir = os.path.abspath(project_dir)
    print(f"Project directory: {project_dir}")
    
    neurox_code_root = find_directory(project_dir, "NeuroXCode")
    if not neurox_code_root:
        print("Error: NeuroXCode directory not found")
        sys.exit(1)
    print(f"NeuroX Code Root: {neurox_code_root}")
    
    evaluation_dir = find_directory(neurox_code_root, "evaluation")
    if not evaluation_dir:
        print("Error: evaluation directory not found")
        sys.exit(1)
    print(f"Evaluation directory: {evaluation_dir}")
    
    data_dir = find_directory(neurox_code_root, "data")
    if not data_dir:
        print("Error: data directory not found")
        sys.exit(1)
    print(f"Data directory: {data_dir}")
    
    alignment_data_dir = os.path.join(data_dir, "alignment_data")
    if not os.path.exists(alignment_data_dir):
        print(f"Error: alignment_data directory not found in {data_dir}")
        sys.exit(1)
    print(f"Alignment data directory: {alignment_data_dir}")
    
    required_files = ["java.in", "java.label", "clusters-500.txt"]
    for file in required_files:
        file_path = os.path.join(alignment_data_dir, file)
        if not os.path.exists(file_path):
            print(f"Error: Required file not found: {file_path}")
        else:
            print(f"Found required file: {file_path}")
    
    os.chdir(evaluation_dir)
    
    if not os.path.exists("alignment.py"):
        print("Error: alignment.py not found in the evaluation directory")
        sys.exit(1)
    print("alignment.py found in the evaluation directory")
    
    # Find activation directory
    activations_dir = os.path.join(project_dir, "NeuroX", "temp", "outputs", "test", "microsoft-codebert-base", "Activations")
    if not os.path.exists(activations_dir):
        print(f"Error: Activations directory not found: {activations_dir}")
        sys.exit(1)
    print(f"Activations directory: {activations_dir}")
    
    # Run the alignment.py script for all layers
    command = [
        sys.executable, "alignment.py",
        "--sentence-file", os.path.join(alignment_data_dir, "java.in"),
        "--label-file", os.path.join(alignment_data_dir, "java.label"),
        "--cluster-file", os.path.join(alignment_data_dir, "clusters-500.txt"),
        "--activation-dir", activations_dir,
        "--thresholds", "90", "95", "100",
        "--methods", "M1", "M2",
        "--output-dir", activations_dir  # Change this line
    ]
    
    try:
        result = subprocess.run(command, check=True)
        print("Alignment completed successfully for all layers.")
    except subprocess.CalledProcessError as e:
        print(f"alignment.py encountered an error: {e}")
        sys.exit(1)
    except FileNotFoundError:
        print(f"Error: alignment.py not found in {evaluation_dir}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Run alignment script for NeuroXCode")
    parser.add_argument("project_dir", help="Path to the LatentConceptAnalysis project directory")
    args = parser.parse_args()
    
    run_alignment(args.project_dir)

if __name__ == "__main__":
    main()