import numpy as np
import scipy.stats as stats


def compute_ranks(errors, n_bootstrap=1000) -> np.ndarray:
    """
    Computes the averaged ranking score in every iteration and for every task..

    :param errors: matrix with M x I x R x N entries, where M are the number of optimizers,
    I are the number of instances or tasks, R is the number of runs per task and
    N are the number of function evaluations per task and run
    :param n_bootstrap: number bootstrap samples to compute the ranks
    :return: the ranks after each iteration as a MxN ndarray, where, as for errors, M are the number of optimizers
    and N are the number of function evaluations
    """
    n_methods = errors.shape[0]
    n_instances = errors.shape[1]
    n_runs = errors.shape[2]
    n_iters = errors.shape[3]

    ranks = np.zeros([n_methods, n_iters])
    for instance_id in range(n_instances):
        for _ in range(n_bootstrap):
            runs = [np.random.randint(n_runs) for i in range(n_methods)]

            rank_samples = [
                stats.rankdata(
                    [errors[i, instance_id, ri, t] for i, ri in enumerate(runs)]
                )
                for t in range(n_iters)
            ]
            ranks += np.array(rank_samples).T
    ranks /= n_instances * n_bootstrap

    return ranks