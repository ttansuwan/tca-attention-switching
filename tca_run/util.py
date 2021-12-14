import numpy as np
from numpy.random import f
import tensortools as tt
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import sys
import copy
import matplotlib 

matplotlib.use('Agg')

def plotting_factors(U, method_name, components, replicate, save_data_dir, marking, seq_trial= None, refitting_per=None):
    # Setting figure and axes
    cmap = mcolors.ListedColormap(["red", "blue"])
    marking_ = copy.deepcopy(marking)

    # 1 and 2 = vertical, 3 and 4 = angled, 1 and 3 = visual, 2 and 4 = olfactory
    vertical_arg = np.where((marking_ == 1) | (marking_ == 2))[0]
    angled_arg = np.where((marking_ == 3) | (marking_ == 4))[0]
    visual_arg = np.where((marking_ == 1) | (marking_ == 3))[0]
    olfactory_arg = np.where((marking_ == 2) | (marking_ == 4))[0]

    marking_[vertical_arg] = 0
    marking_[angled_arg] = 1

    # Figures Setup
    scatter_kw = {'c': copy.deepcopy(marking_), 'cmap': cmap}
    bar_kw = {'width' : 1.0}
    fig, axes = plt.subplots(U.rank, U.ndim, figsize=(12,2*U.rank))
    plt.subplots_adjust(wspace= 0.25, hspace= 0.25, top=0.5, right=0.5)
    # Make axes an array
    if U.rank == 1:
        axes = axes[None, :]
        
    # Reorder Neuron factor from - to +
    U[0] = np.sort(U[0], axis=0, kind='mergesort')
    # Reorder trial factor according to the sequence 
    if seq_trial is not None:
        trial_factor = U[2]
        for i in range(trial_factor.shape[1]):
            factor = trial_factor[:,i]
            trial_factor[:, i] = factor[seq_trial]
        U[2] = trial_factor
    # Run stimuli difference
    angled_vertical_d = angled_vertical_diff(U[2], marking)
    visual_olfactory_d = visual_olfactory_diff(U[2], marking)

    # Plotting factors 
    tt.plot_factors(U, plots=['bar','line','scatter'], fig=fig, axes=axes, scatter_kw=scatter_kw, bar_kw=bar_kw)# plot the low-d factors

    # Use marking_ for vertical/angled visual marking_
    marking_[visual_arg] = 0
    marking_[olfactory_arg] = 1
    visual_stimuli_marking_index = np.split(np.arange(len(marking_)), np.where(np.diff(marking_) != 0)[0]+1)
    
    # Plot the stimuli difference in trial factor plot
    for ax, angled_vertical, visual_olfactory in zip(axes[:,2], angled_vertical_d, visual_olfactory_d):
        ax.text(1.25, 0.5, f's_v:{angled_vertical[0]}\ns_o:{angled_vertical[1]}\ns_t:{visual_olfactory}', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=12)
        for i in visual_stimuli_marking_index:
            temp_marking = marking_[i]
            if 0 in temp_marking:
                ax.axvspan(i[0],i[-1]+1, facecolor='orange', alpha=0.3)
            else:
                ax.axvspan(i[0], i[-1]+1, facecolor='green', alpha=0.3)
    # Plot before/after stimuli onset
    for ax in axes[:,1]:
        ax.axvspan(0, 8, facecolor='purple', alpha=0.3)
        ax.axvspan(8, 17, facecolor='pink', alpha=0.5)

    # Setting axis label
    cols = [f'{col} Factors' for col in ['Neuron', 'Temporal', 'Trial']]
    for ax, col in zip(axes[0], cols):
        ax.set_title(col, fontsize=12)

    cols = ['Neuron', 'Time relative to stimulus onset (s)', 'Trial']
    for ax, col in zip(axes[-1], cols):
        ax.set_xlabel(col, fontsize=12)

    for ax, row in zip(axes[:,0], range(1,U.rank + 1)):
        ax.set_ylabel(f'No. {row}', rotation=0, fontsize=12, labelpad=10)

    # Setting legend - custom legend does not associated with the data (manually edit any changes)
    custom_lines = [Line2D([0], [0], marker='o', color=cmap(0), markersize=4, linewidth=0),
                Line2D([0], [0], marker='o', color=cmap(1), markersize=4, linewidth=0),
                Line2D([0], [0], marker='s', color="orange", markersize=6, linewidth=0),
                Line2D([0], [0], marker='s', color="green", markersize=6, linewidth=0)]
    leg1 = fig.legend(custom_lines, ['Vertical stimulus', 'Angled stimulus', 'Visual block', 'Olfactory block'], bbox_to_anchor=(0.67,1,1,0.4), loc="lower left",ncol=1, fontsize=12)
    fig.add_artist(leg1)
    custom_lines = [Line2D([0], [0], marker='s', color="purple", markersize=6, linewidth=0),
                Line2D([0], [0], marker='s', color="pink", markersize=6, linewidth=0,)]
                
    leg2 = fig.legend(custom_lines, ['Before stimulus onset', 'After stimulus onset'], bbox_to_anchor=(0.37,1,1,0.5), loc="lower left",ncol=1, fontsize=12)

    # Figure
    fig.tight_layout()
    # Save
    if seq_trial is not None:
        if refitting_per is not None:
            path_name = f'{save_data_dir}/{method_name}/refitting/order/{refitting_per}_remove'
        else: 
            path_name = f'{save_data_dir}/{method_name}/order'
    else:
        if refitting_per is not None:
            path_name = f'{save_data_dir}/{method_name}/refitting/no_order/{refitting_per}_remove'
        else:
            path_name = f'{save_data_dir}/{method_name}/no_order'
    if refitting_per is not None:
        path_name = f'{path_name}/method={method_name}-factors_num_components={components+1}-replicate={replicate}-remove_per={refitting_per}.png'
    else:
        path_name = f'{path_name}/method={method_name}-factors_num_components={components+1}-replicate={replicate}.png'
    plt.savefig(path_name, bbox_inches='tight')
    plt.clf()
    plt.close()

def plot_r_squared(r_squared, method_name, no_components, save_data_dir):
    first_quartile = np.percentile(r_squared, 25, axis=1)
    m = np.median(r_squared, axis=1)
    third_quartile = np.percentile(r_squared, 75, axis=1)
    inner_quartile = [m - first_quartile, third_quartile - m]
    
    x = range(1, no_components+1)
    plt.errorbar(x, m, yerr=inner_quartile, fmt='o')
    plt.ylabel('median neuron fit R^2')
    plt.xlabel('no. of components')
    plt.savefig(f'{save_data_dir}/{method_name}/r_2_plot.png')
    plt.clf()

    # test boxplot
    plt.boxplot(r_squared)
    plt.savefig(f'{save_data_dir}/{method_name}/boxplot.png')
    plt.close()

def r_squared(U, V):
    # Mean of org. tensor
    mean_v = np.mean(V)
    # Total sum of squares (TSS)
    squares = (V - mean_v)**2
    TSS = np.sum(squares)
    # Sum of squares of residuals
    squares_residuals = (V - U)**2
    RSS = np.sum(squares_residuals)
    # R-squared
    R_2 = 1 - (RSS/TSS)
    return R_2

def angled_vertical_diff(trial_factor, marking):
    s = []
    # 1 and 2 = vertical, 3 and 4 = angled, 1 and 3 = visual, 2 and 4 = olfactory
    for i in range(trial_factor.shape[1]):
        slice_component = trial_factor[:, i]
        mean_vertical_v = np.mean(slice_component[np.where(marking == 1)])
        mean_angled_v = np.mean(slice_component[np.where(marking == 3)])

        variance_vertical_v = np.var(slice_component[np.where(marking == 1)])
        variance_angled_v = np.var(slice_component[np.where(marking == 3)])
        # Vertical vs Angled in Visual block
        result_v = ((mean_vertical_v - mean_angled_v)**2)/(0.5*(variance_vertical_v+variance_angled_v))

        mean_vertical_o = np.mean(slice_component[np.where(marking == 2)])
        mean_angled_o = np.mean(slice_component[np.where(marking == 4)])

        variance_vertical_o = np.var(slice_component[np.where(marking == 2)])
        variance_angled_o = np.var(slice_component[np.where(marking == 4)])
        # Vertical vs Angled in Olfactory block
        result_o = ((mean_vertical_o - mean_angled_o)**2)/(0.5*(variance_vertical_o+variance_angled_o))     
        s.append((round(result_v, 5), round(result_o,5)))
    return s

def visual_olfactory_diff(trial_factor, marking):
    s = []
    # 1 and 2 = vertical, 3 and 4 = angled, 1 and 3 = visual, 2 and 4 = olfactory
    for i in range(trial_factor.shape[1]):
        slice_component = trial_factor[:, i]
        mean_visual = np.mean(slice_component[np.where((marking == 1) | (marking == 3))])
        mean_olfactory = np.mean(slice_component[np.where((marking == 2) | (marking == 4))])

        variance_visual = np.var(slice_component[np.where((marking == 1) | (marking == 3))])
        variance_olfactory = np.var(slice_component[np.where((marking == 2) | (marking == 4))])
        result_v = np.absolute((mean_visual - mean_olfactory)/np.sqrt((0.5*(variance_visual+variance_olfactory))))
        s.append(round(result_v, 5))
    return s

# def distance_correlation(U_list, factor):
#     U_factor = U_list[:,factor]
#     corr_ = []
#     for i in U_factor:
#         temp_ = []
#         for j in U_factor:
#             distance_cor = dcor.distance_correlation(i, j)
#             temp_.append(distance_cor)
#         corr_.append(temp_)
#     corr_ = np.array(corr_)
#     return corr_