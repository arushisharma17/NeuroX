from setuptools import setup, find_packages

setup(
    name='NeuroXCode',
    version='0.0.1',
    description='NeuroXCode - A Python package for analyzing neural networks and code concepts',
    author='Manjul Balayar, Kellan Bouwman, Akhilesh Nevatia, Ethan Rogers, Sam Frost',
    author_email='',
    url='',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    include_package_data=True,
    install_requires=[
        'dill==0.3.4',
        'imbalanced-learn==0.8.0',
        'matplotlib>=3.7.1',
        'numpy>=1.21.0',
        'pytest==7.0.1',
        'pytest-cov==3.0.0',
        'scikit-learn>=1.0',
        'scipy>=1.7.3',
        'seaborn==0.11.1',
        'sphinx==4.4.0',
        'sphinx-book-theme==0.2.0',
        'svgwrite==1.4.1',
        'tokenizers>=0.12.0',
        'torch>=2.0.0',
        'tqdm>=4.64.1',
        'transformers>=4.12.0',
        'tree_sitter',
        'ufmt==1.3.2',
        'annoy'
    ],
    entry_points={
        'console_scripts': [
            'neuroxcode=NeuroXCode.__main__:main',  # This assumes you have a main entry point in NeuroXCode
        ],
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',  # Adjust to your license
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7',  # Adjust as per the minimum Python version your package supports
)
