# Lobotomy

A framework for two-stage neural architecture search (NAS) and structural pruning on language models.

## Setup

1. clone this repository
2. Install [poetry](https://python-poetry.org/docs/#installation)
3. Make sure to add poetry to your `PATH`
4. `cd` into the directory of the repository
5. Install dependencies with `poetry install`
6. Activate the environment with `poetry shell`

`Note: the library supports and is tested for python 3.9 to 3.11`

```sh
curl -sSL https://install.python-poetry.org | python3 -
git clone git@github.com:aaronkl/lobotomy.git
cd ./lobotomy
poetry install
poetry shell
```

## Structure

```markdown
lobotomy/
├── extract_subnetworks.py
├── __init__.py
├── metrics
│ ├── parameters.py
├── models
│ ├── gpt
│ │ ├── blocks
│ │ │ ├── causal_self_attention.py
│ │ │ ├── __init__.py
│ │ │ ├── mlp.py
│ │ │ └── transformer_block.py
│ │ ├── extract.py
│ │ ├── __init__.py
│ │ ├── model.py
│ │ └── utils.py
│ └── __init__.py
├── modules
│ ├── embedding.py
│ ├── __init__.py
│ ├── layernorm.py
│ ├── linear.py
│ └── rmsnorm.py
├── sampling
│ ├── __init__.py
│ └── random_sampler.py
├── search
│ ├── ask_tell_scheduler.py
│ ├── baselines.py
│ ├── __init__.py
│ ├── local_search.py
│ ├── multi_objective.py
│ └── search.py
├── training_strategies
│ ├── ats.py
│ ├── base_strategy.py
│ ├── __init__.py
│ ├── random_linear.py
│ ├── random.py
│ ├── sandwich.py
│ ├── sandwich_kd.py
│ └── standard.py
└── utils
└── __init__.py
```

