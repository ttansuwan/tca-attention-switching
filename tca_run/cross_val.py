from numba.cuda import test
import numpy as np
import tensortools as tt
import matplotlib.pyplot as plt
import os
import argparse

def perform_cross_val(no_components, method, replication, data, save_dir):
    data_shape = data.shape

    training_error = []
    testing_error = []
    for N in range(replication):
        # Held out 20% of the data randomly for each replication 
        mask = np.random.rand(data_shape[0], data_shape[1], data_shape[2]) > .2
        for R in range(no_components):
            # Fit into the method
            if method == 'ncp_hals':
                U = tt.ncp_hals(data, rank=R+1, mask=mask, verbose=False)
            else:
                U = tt.mcp_als(data, rank=R+1, mask=mask, verbose=False)

            # Compute model prediction for full tensor.
            Xhat = U.factors.full()

            # Compute norm of residuals on training and test sets.
            train_error = np.linalg.norm(Xhat[mask] - data[mask]) / np.linalg.norm(data[mask])
            test_error = np.linalg.norm(Xhat[~mask] - data[~mask]) / np.linalg.norm(data[~mask])

            # Append for plotting
            training_error.append((R, train_error))
            testing_error.append((R, test_error))
    # Converting to np.array for convience 
    training_error = np.array(training_error)
    testing_error = np.array(testing_error)
    plotting_cross_val(training_error, testing_error, method, save_dir)

def plotting_cross_val(training_error, testing_error, method, save_dir):
    # Plotting
    if method == 'ncp_hals':
        title = 'Non-neg fit'
    else:
        title = 'Unconstraint fit'

    fig, (ax1, ax2, ax3) = plt.subplots(1,3, sharey=True, figsize=(10,3)) # Share y axis for interpretability
    fig.suptitle(title, fontsize=16)
    ax1.scatter(training_error[:,0], training_error[:,1], color='blue')
    ax1.plot(np.unique(training_error[:,0]), np.poly1d(np.polyfit(training_error[:,0], training_error[:,1], 2))(np.unique(training_error[:,0])), color='blue')
    ax1.set_title('Training')

    ax2.scatter(testing_error[:,0], testing_error[:,1], color='red')
    ax2.plot(np.unique(testing_error[:,0]), np.poly1d(np.polyfit(testing_error[:,0], testing_error[:,1], 2))(np.unique(testing_error[:,0])), color='red')

    ax2.set_title('Testing')

    ax3.scatter(training_error[:,0], training_error[:,1], color='blue', alpha=0.5)
    ax3.plot(np.unique(training_error[:,0]), np.poly1d(np.polyfit(training_error[:,0], training_error[:,1], 2))(np.unique(training_error[:,0])), color='blue', alpha=0.5)

    ax3.scatter(testing_error[:,0], testing_error[:,1], color='red', alpha=0.5)
    ax3.plot(np.unique(testing_error[:,0]), np.poly1d(np.polyfit(testing_error[:,0], testing_error[:,1], 2))(np.unique(testing_error[:,0])), color='red', alpha=0.5)
    ax3.set_title('Overlaid')

    ax1.set(xlabel='No. of Components', ylabel='Error')
    ax2.set(xlabel='No. of Components', ylabel='Error')
    ax3.set(xlabel='No. of Components', ylabel='Error')
    fig.tight_layout()

    save_dir = f'{save_dir}/{method}/cross_val.png'
    plt.savefig(save_dir)