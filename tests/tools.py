import numpy as np

def add_nans_uniform_everywhere(data, p, rng):
    missing_matrix = rng.choice([np.nan, 1.0], size=data.shape, p=[p, 1-p])
    return data * missing_matrix
