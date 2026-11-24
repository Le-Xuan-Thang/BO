import os

import numpy as np

from benchmarks import sphere
from kernels import kernel_matrix
from sampling import random_sampling

# Problem definition
# call sphere() function
# we can define boundary
# how many dimension we want, let set d=30
objective = sphere
d = 30
bounds_prob = np.array([-100, 100])

lb = np.full(d, bounds_prob[0])
ub = np.full(d, bounds_prob[1])

bounds = np.column_stack((lb, ub))

# Step 2: Algorithm configure, initial sampling and evaluation
n_pop = 10
max_fe = 100
tol = 1e-6
seed = 42

# create x
# initial sampling
X = random_sampling(n_pop, lb, ub)
assert X.shape == (n_pop, d), "please check the X"
print(X.shape)
# initial evaluation

Y = np.array([objective(x) for x in X])
assert Y.shape == (n_pop,), "please check the Y"
# the best solution
best_idx = np.argmin(Y)
best_solution = X[best_idx, :]
print(best_solution)
best_cost = Y[best_idx]
print(best_cost)

# Gaussian Process hyperparameters
length_scale = 1.0
signal_variance = 1.0
noise_variance = 1e-6

# Normalize input variables
X_norm = (X - lb) / (ub - lb)

# Calculate covariance matrix
K = kernel_matrix(X_norm, X_norm, length_scale, signal_variance)

print("Covariance matrix:")
print(K)

print("K shape:", K.shape)
