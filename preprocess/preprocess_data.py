import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
import normalize_methods
import argparse
import sys
import csv

# Export to npy
def normalize_save(rmat, cell_label, save_dir):
    np.save(f"{save_dir}/processed_unnorm.npy", rmat)
    print(f'Saved {save_dir}/processed_unnorm.npy')
    
    norm_rmat = normalize_methods.rescale(rmat)
    np.save(f'{save_dir}/processed_rescale01.npy', norm_rmat)
    print(f'Saved {save_dir}/processed_rescale01.npy')

    shift_rmat = normalize_methods.shifting_min(rmat)
    np.save(f'{save_dir}/processed_shifting.npy', shift_rmat)
    print(f'Saved {save_dir}/processed_shifting.npy')

    zscore_rmat = normalize_methods.z_score(rmat)
    np.save(f'{save_dir}/processed_zscore.npy', zscore_rmat)
    print(f'Saved {save_dir}/processed_zscore.npy')

    np.save(f'{save_dir}/cell_label.npy', cell_label)
    print(f'Saved {save_dir}/cell_label.npy')

def outlier_methods_switch(option):
    if option == 'var':
        detect_outlier_method = normalize_methods.variance_neuron_rank
        threshold_method = normalize_methods.IQR
    elif option == 'std-mean':
        detect_outlier_method = normalize_methods.std_mean_neuron_rank
        threshold_method = normalize_methods.IQR
    elif option == 'snr':
        detect_outlier_method = normalize_methods.signal_to_noise_ratio
        threshold_method = normalize_methods.lower_boundary_threshold
    return detect_outlier_method, threshold_method

def outlier_process(rmat, outlier_name, detect_outlier_method, cell_label, threshold_method, percentage, marking, filepath):
    if outlier_name != 'snr':
        rank = detect_outlier_method(rmat)
        outlier_neuron_no = threshold_method(rank)
    else:
        rank = detect_outlier_method(rmat, marking)
        outlier_neuron_no = threshold_method(rank, int(percentage))

    outlier_neuron_no = np.array(outlier_neuron_no, dtype=int) # Fixing float number error
    remove_outlier_rmat = np.delete(rmat, outlier_neuron_no, axis=0)
    remove_cell_label = np.delete(cell_label, outlier_neuron_no)
    # Save .npy
    normalize_save(remove_outlier_rmat, remove_cell_label, filepath)

def run(mouse_no, detect_outlier_method=None, percentage=0):
    # Generating folders
    files = list(filter(os.path.isdir, os.listdir(f'{os.getcwd()}/matlab_result')))

    mat_folders = os.listdir('./matlab_result')
    matching = [mouse for mouse in mat_folders if mouse_no in mouse]
    if not matching:
        sys.exit('Cannot find the mouse in the matlab_result folder') 

    outlier_methods = ['var', 'std-mean', 'snr']
    mouse_file_path = f'./processed/{matching[0]}'

    filepath = ''
    if detect_outlier_method is not None:
        if detect_outlier_method not in outlier_methods:
            sys.exit('Incorrect input for outlier method')
        else:
            outlier_name = detect_outlier_method
            if detect_outlier_method != 'snr':
                filepath = f'{mouse_file_path}/result/{detect_outlier_method}'
            else:
                filepath = f'{mouse_file_path}/result/{detect_outlier_method}-{percentage}'
            isDetectOutlier = True
    else:
        filepath = f'{mouse_file_path}/result/default'
        isDetectOutlier = False

    if not os.path.exists(filepath):
        os.makedirs(filepath)

    # Read data 
    f = h5py.File(f'./matlab_result/{matching[0]}/neural_activity_concat_mouse1_permute.mat', 'r') # loading fully concatenate (rel and irrel) for mouse 1
    rmat = f[list(f.keys())[-1]] 
    list(f.keys()), rmat

    rmat_act = np.array(rmat)# Convert to np.array
    print("Amount of nan/non-nan for unmasked rmat", np.count_nonzero(np.isnan(rmat_act)), np.count_nonzero(~np.isnan(rmat_act)))

    # Masking time sample - K
    masked_rmat_act = rmat_act[:,8:25,:]  # Mask time-samples before and following 'onset of corridor' (check matlab for the coordinates)
    print('shape', masked_rmat_act.shape)
    print("Amount of nan/non-nan for masked", np.count_nonzero(np.isnan(masked_rmat_act)), np.count_nonzero(~np.isnan(masked_rmat_act)))

    # Find non-nan trial
    full_, nan_trial_no = normalize_methods.find_nan_trial(masked_rmat_act) #w/ masked time sample
    print('w/ masked nan trial no. + no. of nan', nan_trial_no)
    trial_no = np.ones(masked_rmat_act.shape[2], dtype=bool)
    trial_no[nan_trial_no] = False

    # Saving trial sequence
    olfactory_path = f'matlab_result/{matching[0]}/seq_trial_olfactory.mat'
    visual_path = f'matlab_result/{matching[0]}/seq_trial_visual.mat'

    # Order 
    seq_trial, sorted_stimuli_mark, marking  = normalize_methods.order_by_trial(masked_rmat_act, olfactory_path, visual_path, trial_no)

    # Remove trial nan in marking + rmat
    masked_rmat_act_non_nan = masked_rmat_act[:,:,trial_no]

    np.save(f'{mouse_file_path}/index_seq_trial.npy', seq_trial)
    np.save(f'{mouse_file_path}/sorted_stimuli_mark.npy', sorted_stimuli_mark)
    np.save(f'{mouse_file_path}/stimuli_mark.npy', marking)

    # Cell label
    cell_label_path = f'matlab_result/{matching[0]}/cell_label.mat'
    f_cell_label = h5py.File(cell_label_path, 'r') 
    cell_label = f_cell_label[list(f_cell_label.keys())[-1]]
    cell_label = np.array(cell_label[0])

    if isDetectOutlier:
        detect_outlier_method, threshold_method = outlier_methods_switch(detect_outlier_method)
        # No-Order
        print('\nNo Order - detect outlier')
        outlier_process(masked_rmat_act_non_nan, outlier_name, detect_outlier_method, cell_label, threshold_method, percentage, marking, filepath)
    else:
        normalize_save(masked_rmat_act_non_nan, cell_label, filepath)
        with open('./processed/mice_dimensions.csv', mode='a') as csv_file:
            mouse_shape = masked_rmat_act_non_nan.shape
            csv_file.write(f'{mouse_no},{mouse_shape[0]},{mouse_shape[1]},{mouse_shape[2]}\n')