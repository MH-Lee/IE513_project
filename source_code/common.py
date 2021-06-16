import itertools
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# change Matlab aesthetics
mpl.rc('xtick', labelsize=8)
mpl.rc('ytick', labelsize=8)
mpl.rc('text', usetex=False)

# A list of colors and markers that we will iterate through
colors = itertools.cycle(['#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#a65628', '#984ea3', '#999999', '#e41a1c', '#dede00'])
markers = itertools.cycle([ 'h', '*', '<', 'o', 's', 'v', 'D' ])

# The following functions will be of use in transforming the x and y axes in our plots
def log10(x):
    return np.log(x) / np.log(10)

def plot_figure(xtitle, ytitle, x, y, xticks, save_path, ylabels=None, x_transform=log10):
    fig = plt.figure(figsize=(10, 10))
    ax = plt.subplot(111)
    ax.set_xlabel(xtitle,fontsize=12)
    ax.set_xticks(log10(x))
    ax.set_xticklabels(xticks)
    ax.set_ylabel(f"log({ytitle})", fontsize=12)
        
    if ylabels is not None:
        for t in range(len(ylabels)):
            color = next(colors)
            marker = next(markers)
            ax.plot(x_transform(x), log10(y[t]), markersize=4, linestyle="-", marker=marker, color=color,label=ylabels[t], linewidth=1)
        ax.legend(fontsize=10)
    else:
        color = next(colors)
        marker = next(markers)
        ax.plot(x_transform(x), log10(y), markersize=4, linestyle="-", marker=marker, color=color, linewidth=1)
    plt.savefig(save_path)
    plt.show()

def plot_slices(model, n_slices, data, labels, device, save_path, **flags):    
    scale = 1.2 # how wider than the outer sphere do we want the plots to be?
    x_min, x_max = -scale * flags["outer_radius"], scale * flags["outer_radius"]
    y_min, y_max = -scale * flags["outer_radius"], scale * flags["outer_radius"]
    
    # create a grid of points to evaluate the output at
    grid = 64
    h = (x_max - x_min) / float(grid)
    xx, yy = torch.meshgrid(torch.arange(x_min, x_max, h),
                            torch.arange(y_min, y_max, h))
    
    # pick pairs of random indices from the dataset
    random_indices = np.random.choice(range(data.shape[0]), size=2 * n_slices, replace=False)
    main_fig, main_ax = plt.subplots(1, n_slices, sharex='col', sharey='row', figsize=(3 * n_slices, 3))
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(model.parameters(), lr=flags["learning_rate"])
    
    for t in range(n_slices):
        i = random_indices[2 * t]
        j = random_indices[2 * t + 1]
        
        # this is an array of 2d datapoints
        contour_data_2d = torch.tensor(np.c_[xx.ravel(), yy.ravel()])
        
        # x_vec and y_vec correspond to two orthogonal directions
        # the plane spanned by which contains the two chosen datapoints
        x_vec = data[i, :]
        y_vec = data[j, :]
        y_vec = y_vec - np.dot(x_vec, y_vec) * x_vec / np.linalg.norm(x_vec)
        y_vec = y_vec / (np.linalg.norm(y_vec))
        x_vec = x_vec / (np.linalg.norm(x_vec))
        
        # This projects the 2d grid onto the high dimensional space,
        # so that it can be input to the neural network
        contour_data = np.matmul(contour_data_2d, torch.vstack([x_vec, y_vec]))
        tensor_data = TensorDataset(contour_data, \
                                    torch.tensor(np.zeros(shape=(contour_data.shape[0],2))))
        # pass the grid through the network
        outputs = perform_op_over_data(model, tensor_data, 
                                       criterion, optimizer,
                                       eval=True,
                                       device=device, batch_size=flags["batch_size"])

        # plot a contour
        ax = main_ax[t]
        Z = np.array([])
        for output in outputs:
            Z = np.concatenate([Z,output[:,0]-output[:,1] > 0])
        Z = Z.reshape(xx.shape)
        ax.contourf(xx, yy, Z, cmap=plt.get_cmap("Paired"))
        ax.axis('off')
        ax.set_ylim([y_min,y_max])
        ax.set_xlim([x_min,x_max])

        # mark the two datapoints on this contour
        for ind in [i,j]:
            if labels[ind,0]==1:
                marker = 'o'
            else:
                marker = '*'
            ax.scatter(np.dot(data[ind, :],x_vec), np.dot(data[ind, :],y_vec), 
                       c='black', s=30, marker=marker, cmap=plt.cm.Paired)
        # add the cirles
        circle1=plt.Circle((0, 0), 1, color='black',fill=False, linewidth=1)
        circle2=plt.Circle((0, 0), flags["outer_radius"], color='black',fill=False, linewidth=1)
        ax.add_artist(circle1)
        ax.add_artist(circle2)
        ax.set(aspect='equal')
    plt.savefig(save_path)
    plt.show()
    
def perform_op_over_data(model, dataset, criterion, optimizer, device, batch_size, eval=False):
    outs = []
    dataloader = DataLoader(dataset, batch_size=batch_size, pin_memory=True)
    
    if eval:
        model.eval()
    else:
        model.train()
        
    for idx, (inputs, labels) in tqdm(enumerate(dataloader), leave=False):
        inputs, labels = inputs.to(device), labels.to(device)
        if eval == False:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                outputs = model(inputs) 
        outs.append(outputs.detach().cpu())
    return outs

def get_margin_error(model, dataset, criterion, optimizer, device, batch_size, margin=0.0, median=False, mode='synthetic'):
    logits = perform_op_over_data(model, dataset, criterion, optimizer, device, eval=True, batch_size=batch_size)
    logits = torch.vstack(logits)
    labels = dataset. tensors[1]
    if (mode == 'synthetic') or ('bin' in mode):
        print("binary")
        label_ind = torch.argmax(labels, axis=1) # Indices of correct labels
#     elif mode == 'mnist_bin':
#         label_ind = torch.argmax(labels, axis=1) # Indices of correct labels
#     elif mode == 'fmnist_bin':
#         label_ind = torch.argmax(labels, axis=1) # Indices of correct labels
    else:
        print("multi-output")
        label_ind = labels

    # The following code finds the maximum output of the network after ignoring the true class
    modified_logits = logits.clone().detach()
    for i in range(logits.size(0)):
        modified_logits[i, label_ind[i]] = -float("inf") 

    max_wrong_logits = torch.max(modified_logits, axis=1)
    max_true_logits = logits[[i for i in range(logits.size(0))], list(label_ind)]   # The output of the network on the true class

    if median:
        error = torch.median(((max_true_logits - max_wrong_logits[0]) < float(margin)).float())
    else:
        error = torch.mean(((max_true_logits - max_wrong_logits[0]) < float(margin)).float()) 
    return error