# Lobotomy
A framework for two-stage neural architecture search (NAS) and structural pruning on language models.

## Setup

Create a virtual environment if you are not using one. Using miniconda:

```sh
$ conda create -n lobotomy python==3.9
$ conda activate lobotomy
$ pip install --upgrade pip setuptools
```

To install in developer mode (-e) run:

```sh
$ git clone https://github.com/aaronkl/lobotomy
$ cd lobotomy
$ pip install -e .[all]
```

For a minimal installation use `pip install -e .` instead.

## Structure

```markdown
lobotomy/
├── extract_subnetworks.py
├── __init__.py
├── metrics
│   ├── magnitude.py
│   └── parameters.py
├── models
│   ├── gpt
│   │   ├── blocks
│   │   │   ├── causal_self_attention.py
│   │   │   ├── __init__.py
│   │   │   ├── mlp.py
│   │   │   └── transformer_block.py
│   │   ├── extract.py
│   │   ├── __init__.py
│   │   ├── model.py
│   │   └── utils.py
│   └── __init__.py
├── modules
│   ├── embedding.py
│   ├── __init__.py
│   ├── layernorm.py
│   ├── linear.py
│   └── rmsnorm.py
├── sampling
│   ├── __init__.py
│   └── random_sampler.py
├── search
│   ├── ask_tell_scheduler.py
│   ├── baselines.py
│   ├── __init__.py
│   ├── local_search.py
│   ├── multi_objective.py
│   └── search.py
├── training_strategies
│   ├── ats.py
│   ├── base_strategy.py
│   ├── __init__.py
│   ├── random_linear.py
│   ├── random.py
│   ├── sandwich.py
│   └── standard.py
└── utils
    ├── auto_search_config.py
    └── __init__.py
```

