import os
import argparse
# from .extract_activations import main as extract_activations
from NeuroXCode.process_activations import extract_activations

def setup_environment(project_dir):
    os.environ['PROJECTDIR'] = project_dir
    os.environ['MAMBA_ROOT_PREFIX'] = os.path.join(project_dir, 'micromamba')
    os.environ['NEUROX_DIR'] = os.path.join(project_dir, 'NeuroX')

def main():
    parser = argparse.ArgumentParser(description="Setup environment and run extraction")
    parser.add_argument("project_dir", type=str, help="Path to the project directory")
    args = parser.parse_args()

    setup_environment(args.project_dir)
    extract_activations(args.project_dir)

if __name__ == "__main__":
    main()