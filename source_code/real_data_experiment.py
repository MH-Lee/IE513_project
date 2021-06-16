import torch
import numpy as np
from torch.utils.data import TensorDataset

def dataset_sampling(data, n,  test_sample=False, exclude_sample=None, binary=False):
    if exclude_sample is not None:
        exclude_idx = data.index.isin(exclude_sample)
        data = data[~exclude_idx]
        print(data.shape)
    sample_ratio = (data['label'].value_counts()/data.shape[0] * n).round().sort_index()
    new_sample_data = np.empty((0, data.shape[1]), float)
    sample_idx = list()
    for idx in sample_ratio.index.tolist():
        sample_data = data[data['label'] == idx].sample(n=int(sample_ratio[idx]), replace=False, random_state=8312)
        sample_idx.append(sample_data.index.tolist())
        new_sample_data = np.append(new_sample_data, sample_data.values, axis=0)
    np.random.shuffle(new_sample_data)
    new_sample_data = new_sample_data[:n]
    print("data shape : ", new_sample_data.shape)

    data, labels = new_sample_data[:,1:]/255, new_sample_data[:,0].astype(np.int64)
    labels = torch.tensor(labels, dtype=torch.long)
    if binary:
        num = np.unique(labels, axis=0).shape[0]
        labels = labels.reshape(-1)
        labels = torch.tensor(np.eye(num)[labels], dtype=torch.float)
    if test_sample:
        return_value = (TensorDataset(torch.tensor(data).float(), labels), sample_idx)
    else:
        return_value = TensorDataset(torch.tensor(data).float(), labels)
    return return_value

def get_bounds_mnist(initial_weights, final_weights, spectral_norms, train_samples, **flags):   
    # Neyshabur+ ICLR'17 with distance from initialization (plotted on our paper)
    pac_spectral_bound_1 = np.max(np.linalg.norm(train_samples, axis=1))*flags['depth']*np.sqrt(flags['width'])
    pac_spectral_bound_1 *= np.linalg.norm([np.linalg.norm(final_weights[i]-initial_weights[i])/spectral_norms[i] for i in range(len(final_weights))])
    pac_spectral_bound_1 *= np.prod(spectral_norms)
    
    # Bartlett+'17 bound
    covering_spectral_bound = np.max(np.linalg.norm(train_samples, axis=1))
    covering_spectral_bound *= np.power(np.sum([np.power(np.sum(np.linalg.norm(final_weights[i]-initial_weights[i])) / spectral_norms[i],2.0/3.0) \
                                                for i in range(len(final_weights))]),1.5)
    covering_spectral_bound *= np.prod(spectral_norms)

    # Neyshabur et al '19, ignoring the sqrt h
    # This applies to only 1 hidden layer networks, i.e., depth =1
    unilayer_bound = np.linalg.norm(initial_weights[0],ord=2)*np.linalg.norm(final_weights[1])
    unilayer_bound += np.linalg.norm(final_weights[0]-initial_weights[0])*np.linalg.norm(final_weights[1])
    unilayer_bound *= np.max(np.linalg.norm(train_samples, axis=1))
    return covering_spectral_bound, pac_spectral_bound_1, unilayer_bound