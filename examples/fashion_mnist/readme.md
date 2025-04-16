# Demo

This is a minimal example demonstrating how to pretrain a supernet and subsequently search for optimal subnetworks.
We train a simple model that uses `whittle.modules.Linear` on the FashionMNIST dataset and then perform a multi-objective search over the supernet.

## Phase 1: Train the supernet
To train the supernet, run the following command:

`python train_fashion_mnist.py --training_strategy <train-strategy>`

`<train-strategy>` can be set to one of the following options:

1. `standard`: Train the entire supernet at every optimization step.
2. `sandwich`: At each step, train the largest subnetwork, the smallest subnetwork, and a few randomly sampled subnetworks.
3. `random`: Train randomly sampled subnetworks at every step.
4. `random_linear`: With probability *p*, train a randomly sampled subnetwork; otherwise, train the full supernet. The value of *p* increases linearly from 0 to 1 over time.
5. `ats`: Alternate training between the full supernet (on odd steps) and randomly sampled subnetworks (on even steps).

## Phase 2: Search the trained supernet
To search the trained supernet, run the following command:

`python search_fashion_mnist.py --training_strategy <train-strategy> --search_strategy <search-strategy>`

Replace `<train-strategy>` with the training strategy used during supernet pretraining, and `<search-strategy>` with one of the following options:

1. `random_search`: Purely random search over the configuration space.
2. `stratified_random_search`: Stratified random sampling ensuring uniform coverage of the configuration space.
3. `local_search`: Hill-climbing style local search to refine solutions.
4. `morea`: Multi-objective Regularized Evolutionary Algorithm.
5. `nsga2`: NSGA-II, a fast and elitist multi-objective genetic algorithm [Deb et al., 2002].
6. `lsbo`: Bayesian optimization using a linear scalarization of objectives.
7. `rsbo`: Bayesian optimization with random scalarization of objectives.