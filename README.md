# NeuroX Toolkit

![NeuroX-Workflow](https://github.com/user-attachments/assets/1617c5b0-5a72-4106-9c6b-94c0c8097fef)


### 1. Clone NeuroX

```bash
cd <your project dir>    #eg. cd /work/LAS/jannesar-lab/arushi/LatentConceptAnalysis

# Clone the NeuroX repository
git clone https://github.com/arushisharma17/NeuroX.git
cd NeuroX
```

### 2. Creating the environment 

#### a. For pronto users only: Set Up NeuroX on Pronto using micromamba

```bash
sbatch pronto_setup.sh /path/to/project/dir
```
Important: Please not that if you ran setup.sh while installing CodeConceptNet repo, you do not need to install NeuroX again. The setup script already does that for you. It should be located at your $PROJECTDIR/ . Its an editable install, so you can cd into $PROJECTDIR/NeuroX and git pull to get latest changes. 

For further information about pronto: [Pronto Documentation](https://research.it.iastate.edu/guides/pronto/getting_started/)

#### b. For other (non-pronto) users: Create a python virtual environment

```bash
python3 -m venv neurox-env
source neurox-env/bin/activate
pip install --upgrade pip

# Create a virtual environment
python3 -m venv neurox-env

# Activate the environment
source neurox-env/bin/activate

# Upgrade pip
pip install --upgrade pip

pip install -e . #Installs package in editable mode

```


### 3. Explore functionality of this toolkit: (For Testing and Educational Purposes)
We have created a temp directory where we store all relevant files for testing purposes. 

#### a. If you are on pronto: you can use interactive mode to run scripts. 

```bash
srun --time=01:00:00 --nodes=1 --cpus-per-task=8 --partition=gpu-interactive --gres=gpu:1 --pty /usr/bin/bash
cd temp/
./test_run_interactive.sh /path/to/project/dir   #eg. /work/LAS/jannesar-lab/arushi/LatentConceptAnalysis
```
Now you can run the code that you normally put in your shell script here directly. You will need to activate the environment and load modules too. 
[Pronto Interactive Computing Guide](https://research.it.iastate.edu/guides/pronto/interactive_computing/)

#### b. If you are not on pronto: can use following colab notebook as a reference. 
You can run code interactively in colab or on your own machine after setting up the environment. 

Add link to notebook


### After running extract_activations.py

```
/path/to/your/project_directory/
└── NeuroX/
    └── temp/
        ├── config.json
        ├── input_data/
        │   ├── <dataset1>/                # e.g., test, java, cuda
        │   │   └── <input files>          # e.g., test.in, test.label
        │   ├── <dataset2>/                # e.g., cuda
        │   │   └── <input files>          # e.g., cuda.in cuda.label
        │   └── <dataset3>/                # e.g., java
        │       └── <input files>          # e.g., java.in java.label 
        ├── outputs/
        │   ├── <dataset1>/                # e.g., test
        │   │   ├── <model1>/              # e.g., microsoft-codebert-base
        │   │   │   └── Activations/
        │   │   │       ├── layer0/
        │   │   │       ├── layer1/
        │   │   │       └── layerN/
        │   │   └── <model2>/              # e.g., bert-base-cased
        │   │       └── Activations/
        │   │           ├── layer0/
        │   │           ├── layer1/
        │   │           └── layerN/
        │   └── <dataset2>/                # e.g., java
        │       ├── <model1>/              # e.g., microsoft-codebert-base
        │       │   └── Activations/
        │       │       ├── layer0/
        │       │       ├── layer1/
        │       │       └── layerN/
        │       └── <model2>/              # e.g., bert-base-cased
        │           └── Activations/
        │               ├── layer0/
        │               ├── layer1/
        │               └── layerN/
        └── test_run_interactive.sh

```










<p align="center">
  <img src="https://github.com/fdalvi/NeuroX/raw/master/docs/intro/logo.png" />
</p>

[![Tests Runner](https://github.com/fdalvi/NeuroX/actions/workflows/python-package.yml/badge.svg)](https://github.com/fdalvi/NeuroX/actions/workflows/python-package.yml)
[![API Reference](https://img.shields.io/badge/API-Reference-blue)](https://neurox.qcri.org/docs/)

NeuroX provide all the necessary tooling to perform Interpretation and Analysis of (Deep) Neural Networks centered around _Probing_. Specifically, the toolkit provides:

- Support for extraction of activation from popular models including the entirety of [transformers](https://github.com/huggingface/transformers), with extended support for other models like [OpenNMT-py](https://github.com/OpenNMT/OpenNMT-py) planned in the near future
- Support for training linear probes on top of these activations, on the entire activation space of a model, on specific layers, or even on specific set of neurons.
- Support for neuron extraction related to specific concepts, using the _Linear Correlation Analysis_ method ([What is one Grain of Sand in the Desert? Analyzing Individual Neurons in Deep NLP Models.](https://ojs.aaai.org/index.php/AAAI/article/view/4592/4470)). The toolkit can extract either a local ranking of neurons important to a particular target class, or a global ranking of neurons important to all the target classes.
- Support for ablation analysis by either removing or zeroing out specific neurons to determine their function and importance.
- Support for subword and character level aggregation across a variety of tokenizers, including BPE and all tokenizers in the [transformers](https://github.com/huggingface/transformers) library.
- Support for activation visualization over regular text, to generate qualitative samples of neuron activity over particular sentences.

A demo using a lot of functionality provided by this toolkit [is available](https://neurox.qcri.org/demo).

![demo picture](https://github.com/fdalvi/NeuroX/raw/master/docs/intro/en-ar-NeuroX-demo.png)

## Getting Started
This toolkit requires and is tested on Python versions 3.6 and above. It may work with older Python versions with some fiddling, but is currently not tested nor supported. The easiest way to get started is to use the published pip package:

```bash
pip install neurox

```

#### Manual Installation
If you wish to install this package manually (e.g. to modify or contribute to the code base), you can clone this repository into a directory of your choice:

```bash
git clone https://github.com/fdalvi/NeuroX.git
```

Create and activate a new virtual environment for the toolkit (This step can be skipped if you manage your environment in other ways like Conda or System-level installations):

```bash
python -m venv .neurox-env
source .neurox-env/bin/activate
```

Install the dependencies required to run the toolkit:

```bash
pip install -e .
```

## Sample Code
A Jupyter notebook with a complete example of extracting activations from BERT, training a toy task, extracting neurons and visualizing them is available in the [examples](examples/End%20to%20End%20Example.ipynb) directory for a quick introduction to the main functionality provided by this toolkit.

## Documentation
[API Reference](https://neurox.qcri.org/docs/) contains an API reference for all of the functions exposed by this toolkit. Primarily, the toolkit's functionality is separated into several high-level components:

- Extraction
- Data Preprocessing
- Linear Probing
- Neuron extraction and interpretation
- Neuron cluster analysis
- Visualization

## Citation

Please cite our paper published at AAAI'19 if you use this toolkit.

```
@article{dalvi2019neurox,
  title={NeuroX: A Toolkit for Analyzing Individual Neurons in Neural Networks},
  author={Dalvi, Fahim
    and Nortonsmith, Avery
    and Bau, D Anthony
    and Belinkov, Yonatan
    and Sajjad, Hassan
    and Durrani, Nadir
    and Glass, James},
  journal={Proceedings of the AAAI Conference on Artificial Intelligence (AAAI)},
  year={2019}
}
```

## Planned features

- [x] Pip package
- [ ] Support for OpenNMT-py models
- [ ] Support for control tasks and computing metrics like selectivity
- [ ] Support for attention and other module analysis

## Publications
- Hassan Sajjad, Narine Kokhlikyan, Fahim Dalvi, Nadir Durrani (2021). [Fine-grained Interpretation and Causation Analysis in Deep NLP Models.](https://www.aclweb.org/anthology/2021.naacl-tutorials.2.pdf) In Proceedings of the 18th Annual Conference of the North American Chapter of the Association of Computational Linguistics: Human Language Technologies (NAACL), Virtual, June
- Nadir Durrani, Hassan Sajjad, Fahim Dalvi (2021). [How transfer learning impacts linguistic knowledge in deep NLP models?](https://alt.qcri.org/~ndurrani/pubs/ACL-FT-2021.pdf) In Findings of the Association for Computational Linguistics (ACL-IJCNLP). Virtual, August
- Yonatan Belinkov\*, Nadir Durrani\*, Fahim Dalvi, Hassan Sajjad, Jim Glass (2020). [On the Linguistic Representational Power of Neural Machine Translation Models.](https://www.aclweb.org/anthology/2020.cl-1.1.pdf) Computational Linguistics. 46(1), pages 1 to 57 (\*Equal Contribution––Alphabetic Order).
- Nadir Durrani, Hassan Sajjad, Fahim Dalvi, Yonatan Belinkov (2020). [Analyzing Individual Neurons in Pre-trained Language Models.](https://www.aclweb.org/anthology/2020.emnlp-main.395.pdf) In Proceedings of the 17th Conference on Empirical Methods in Natural Language Processing (EMNLP), Punta Cana, Dominican Republic, November.
- Fahim Dalvi, Hassan Sajjad, Nadir Durrani, Yonatan Belinkov (2020). [Analyzing Redundancy in Pretrained Transformer Models](https://www.aclweb.org/anthology/2020.emnlp-main.398.pdf). In Proceedings of the 17th Conference on Empirical Methods in Natural Language Processing (EMNLP), Punta Cana, Dominican Republic, November.
- John M Wu\*, Yonatan Belinkov\*, Hassan Sajjad, Nadir Durrani, Fahim Dalvi and James Glass (2020). [Similarity Analysis of Contextual Word Representation Models.](https://www.aclweb.org/anthology/2020.acl-main.422.pdf) In Proceedings of the 58th Annual Conference of the Association for Computational Linguistics (ACL). Seattle, USA, July (\*Equal Contribution––Alphabetic Order).
- Anthony Bau\*, Yonatan Belinkov\*, Hassan Sajjad, Fahim Dalvi, Nadir Durrani, and James Glass (2019). [Identifying and Controlling Important Neurons in Neural Machine Translation.](https://openreview.net/pdf?id=H1z-PsR5KX) In Proceedings of the 7th International Conference on Learning Representations (ICLR). New Orleans, USA, May (\*Equal Contribution––Alphabetic Order).
- Nadir Durrani, Fahim Dalvi, Hassan Sajjad, Yonatan Belinkov, and Preslav Nakov (2019). [One Size Does Not Fit All: Comparing NMT Representations of Different Granularities.](https://www.aclweb.org/anthology/N19-1154.pdf) In Proceedings of the 17th Annual Conference of the North American Chapter of the Association of Computational Linguistics: Human Language Technologies (NAACL), Minneapolis, US, June
- Fahim Dalvi\*, Nadir Durrani\*, Hassan Sajjad\*, Yonatan Belinkov, D. Anthony Bau, and James Glass (2019). [What is one Grain of Sand in the Desert? Analyzing Individual Neurons in Deep NLP Models.](https://ojs.aaai.org/index.php/AAAI/article/view/4592/4470) In Proceedings of the 33rd AAAI Conference on Artificial Intelligence (AAAI). Honolulu, USA, Jan. (\*Equal Contribution––Alphabetic Order).
- Yonatan Belinkov, Nadir Durrani, Fahim Dalvi, Hassan Sajjad, James Glass (2017). [What do Neural Machine Translation Models Learn about Morphology?](https://www.aclweb.org/anthology/P17-1080.pdf) In Proceedings of the 55th Annual Conference of the Association for Computational Linguistics (ACL), Vancouver, Canada, July.
- Fahim Dalvi, Nadir Durrani, Hassan Sajjad, Yonatan Belinkov and Stephan Vogel (2017). [Understanding and Improving Morphological Learning in the Neural Machine Translation Decoder.](https://www.aclweb.org/anthology/I17-1015.pdf) In Proceedings of the 8th International Conference on Natural Language Processing (IJCNLP), Taipei, Taiwan, November.
- Yonatan Belinkov, Lluís Màrquez, Hassan Sajjad, Nadir Durrani, Fahim Dalvi and James Glass (2017). [Evaluating Layers of Representation in Neural Machine Translation on Part-of-Speech and Semantic Tagging Tasks.](https://www.aclweb.org/anthology/I17-1001.pdf) In Proceedings of the 8th International Conference on Natural Language Processing (IJCNLP), Taipei, Taiwan, November
