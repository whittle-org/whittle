# whittle

Whittle is a library for Neural Architecture Search (NAS) aimed at compressing large language models (LLMs). The core idea is to treat the network as a super-network and select sub-networks that optimally balance performance and efficiency. Whittle provides the following functionalities:

- Support for a variety of LLMs to define super-networks
- Checkpoint generation for sub-networks, enabling deployment
- Downstream evaluation of sub-networks using LM-Eval-Harness
- A simple interface for various super-network training strategies and multi-objective search methods

## Installation

Whittle supports and is tested for python 3.9 to 3.11. 

You can install whittle with: 
```
pip install whittle
```


### Install from source  

Install whittle from source to get the most recent version:
```
git clone git@github.com:whittle-org/whittle.git
cd whittle
pip install -e .
```
### Getting started with whittle  

To explore and understand different functionalities of ```whittle``` checkout [this](https://colab.research.google.com/drive/1xFhsHrqJGQnFuigLCqKHsJFLsJwksl4v?usp=sharing) colab notebook and ```examples/```

## Projects that use whittle

- [Structural Pruning of Pre-trained Language Models via Neural Architecture Search](https://github.com/whittle-org/plm_pruning)
- [HW-GPT Bench](https://github.com/automl/HW-GPT-Bench)

## How to get involved

We more than happy for any code contribution. If you are interested in contribution to whittle, 
please read our [contribution guide](CONTRIBUTING.md).