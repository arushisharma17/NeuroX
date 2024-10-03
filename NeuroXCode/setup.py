from setuptools import setup, find_packages

# Read the contents of your README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="neuroxcode",
    version="0.0.1",
    author="Manjul Balayar, Kellan Bouwman, Akhilesh Nevatia, Ethan Rogers, Sam Frost",
    author_email="",
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
    packages=find_packages(where="src"),  # Ensures all modules in src/ are included
    package_dir={"": "src"},  # Defines src/ as the root for the package
    python_requires=">=3.10",
    install_requires=[
        "imbalanced-learn==0.8.0",
        "numpy>=1.21.0",
        "scikit-learn>=1.0",
        "scipy>=1.7.3",
        "seaborn==0.11.1",
        "svgwrite==1.4.1",
        "transformers>=4.12.0",
        "tokenizers>=0.12.0",
        "torch>=2.0.0",
        "matplotlib>=3.7.1",
        "tqdm>=4.64.1",
        "dill==0.3.4",
        "build==0.7.0",
        "pytest==7.0.1",
        "pytest-cov==3.0.0",
        "sphinx==4.4.0",
        "sphinx-book-theme==0.2.0",
        "ufmt==1.3.2",
        "tree_sitter",
    ],
    entry_points={
        'console_scripts': [
            'neuroxcode=neuroxcode.__main__:main',
        ],
    },
    include_package_data=True,
)
