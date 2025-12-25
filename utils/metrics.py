"""
MIT License

Copyright (c) 2025 Zhenyi Zhao
Copyright (c) University of Dundee
"""


import numpy as np
from sklearn.utils import resample


def bootstrap(seed, n_iterations, alpha, dice):
    seeds = np.random.RandomState(seed)
    bootstrap_means = []
    
    for i in range(n_iterations):
        bootstrap_sample = resample(dice, n_samples=len(dice), random_state=seeds)
        bootstrap_means.append(np.mean(bootstrap_sample))
    
    lower = ((1.0-alpha)/2.0)*100
    upper = (alpha+((1.0-alpha)/2.0))*100

    ci = np.percentile(bootstrap_means, [lower, upper])

    return ci