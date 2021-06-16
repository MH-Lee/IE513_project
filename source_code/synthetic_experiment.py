import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from tqdm.auto import tqdm

def get_random_data(n_points, input_dim=10, inner_radius=1, outer_radius=1.1):
    data = torch.randn(size=(n_points, input_dim)) 
    labels = torch.zeros(size=(n_points, 2))

    for i in range(n_points):
        if torch.rand(size=(1,)) > 0.5:
            labels[i, 0] = 1
            data[i, :] = inner_radius * (data[i, :] / torch.linalg.norm(data[i, :]))
        else:
            labels[i, 1] = 1
            data[i, :] = outer_radius * data[i, :] / torch.linalg.norm(data[i, :])    
    return TensorDataset(data, labels)

def get_bounds(initial_weights, final_weights, spectral_norms, n_train, **flags):
    bound = spectral_norms[0] * spectral_norms[2]
    bound /= np.sqrt(n_train)
    bound /= flags['margin']
    bound *= np.power(np.sum([np.power(np.linalg.norm(np.sum(np.linalg.norm(final_weights[k] - initial_weights[k], axis=0)))
                                       / spectral_norms[k], 2.0 / 3.0) 
                                       for k in [0, 2]]), 1.5) 
    
    bound_0 = bound # Bartlett et al., bound

    bound = spectral_norms[0] * spectral_norms[2]
    bound /= np.sqrt(n_train)
    bound /= flags['margin']
    bound *= flags["depth"]*np.sqrt(flags["width"])
    bound *= np.power(np.sum([np.power(np.linalg.norm(final_weights[k] - initial_weights[k])
                                       / spectral_norms[k], 2)
                                       for k in [0, 2]]), 0.5)
    bound_1 = bound # Neyshabur et al., '18 bound
    
    # WARNING: This only holds for 2 layer networks
    bound = 1.0
    bound /= np.sqrt(n_train)
    bound /= flags['margin']
    bound *= np.linalg.norm(final_weights[2])
    bound *= (np.linalg.norm(final_weights[0] - initial_weights[0]) + np.linalg.norm(initial_weights[0], ord=2))
    bound_2 = bound # Neyshabur et al., '19 bound
    return bound_0, bound_1, bound_2