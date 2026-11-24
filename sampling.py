import numpy as np


def random_sampling(n_samples, lb, ub):
    d = len(lb)
    rng = np.random.default_rng(d)
    # this equal to X = lb + rng*(ub-lb)
    X = rng.uniform(low=lb, high=ub, size=(n_samples, d))
    return X
