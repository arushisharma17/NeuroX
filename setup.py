from setuptools import setup, find_packages

# Read the contents of your README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="neurox",
    version="1.0.9",
    author="Arushi Sharma",
    author_email="arushi17@iastate.edu",
    description="Toolkit for Neuron Analysis in Deep NLP Models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/arushisharma17/NeuroX.git",
    project_urls={
        "Documentation": "https://neurox.qcri.org/docs/",
        "Bug Tracker": "https://github.com/arushisharma17/NeuroX/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    packages=find_packages(where="."),
    install_requires=[
        "h5py==3.6.0",
        "imbalanced-learn==0.8.0",
        "numpy>=1.21.0",
        "scikit-learn>=1.0",
        "scipy>=1.7.3",
        "seaborn==0.11.1",
        "svgwrite==1.4.1",
        "transformers>=4.12.0",  # Update transformers to a compatible version
        "tokenizers>=0.12.0",  # Ensure this is compatible
        "torch>=2.0.0",
        "matplotlib>=3.7.1",
        "tqdm>=4.64.1",
        "seaborn==0.11.1",
        
    ],
    python_requires=">=3.6",
    extras_require={
        "dev": [
            "build==0.7.0",
            "pytest==7.0.1",
            "pytest-cov==3.0.0",
            "sphinx==4.4.0",
            "sphinx-book-theme==0.2.0",
            "ufmt==1.3.2",
        ]
    },
)
