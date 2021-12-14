import numpy as np
import h5py

# Normalized between 0-1
def rescale(rmat):
    min_x = np.min(rmat)
    max_min_x = np.max(rmat) - min_x
    norm_rmat = (rmat - min_x)/ max_min_x
    return norm_rmat

# Shifting to non-negative by adding the minimum number
def shifting_min(rmat):
    min_x = np.min(rmat)*-1
    norm_rmat = rmat + min_x
    return norm_rmat

# Z-score 
def z_score(rmat):
    mean_x = np.mean(rmat)
    std_x = np.std(rmat)
    norm_rmat = (rmat - mean_x)/std_x
    return norm_rmat

# Variance
def variance_neuron_rank(input):
    rank = []
    for i in range(input.shape[0]):
        slice_neuron = input[i]
        unfold_slice_neuron = slice_neuron.flatten()
        var = np.var(unfold_slice_neuron)
        mean = np.average(unfold_slice_neuron)
        rank.append((i, var, mean))
    rank = np.array(sorted(rank, reverse=True, key=lambda pair: pair[1]))
    return rank

# Coefficient of variation
def std_mean_neuron_rank(input):
    rank = []
    for i in range(input.shape[0]):
        slice_neuron = input[i]
        unfold_slice_neuron = slice_neuron.flatten()
        std = np.std(unfold_slice_neuron)
        mean = np.average(unfold_slice_neuron)
        rank.append((i, std/mean))
    rank = np.array(sorted(rank, reverse=True, key=lambda pair: pair[1]))
    return rank

# Signal to noise ratio
def signal_to_noise_ratio(rmat, marking):
    snr_list = []
    for i in range(rmat.shape[0]):
        signals = []
        for j in range(4):  # 4 condition stimuli
            marks = np.where(marking == j+1)[0]
            values = rmat[i,:,marks[0]:marks[-1]+1] # At ith neuron, all timesample, at marking trial
            timesample_mean = np.mean(values, axis=1) # Compute mean of each timesample (17,)
            signals.append(timesample_mean)
        signals_var = np.var(signals)
        total_var = np.var(rmat[i])
        snr = signals_var/total_var # signal/noise
        snr_list.append([i, snr])
    rank = np.array(sorted(snr_list, reverse=True, key=lambda pair: pair[1]))
    return rank

# Limit variance with IQR 
def IQR(rank):
    Q1 = np.percentile(rank[:,1], 25)
    Q3 = np.percentile(rank[:,1], 75)
    IQR = Q3 - Q1
    min_iqr = Q1 - (1.5 * IQR)
    max_iqr = Q3 + (1.5 * IQR)
    print(f'Q1 = {Q1} Q3={Q3}, min_iqr = {min_iqr}, max_iqr = {max_iqr}, iqr = {IQR}')

    outlier_mask = np.where((rank[:,1] > max_iqr) | (rank[:,1] < min_iqr), True, False)
    outlier_mask_lower = np.where(rank[:,1] < min_iqr, True, False)
    outlier_mask_upper = np.where(rank[:,1] > max_iqr, True, False)

    outlier_neuron_no = rank[outlier_mask]
    outlier_per = f'Overall outlier percentage: {(np.sum(outlier_mask)/len(rank))*100:.3f}% Upper IQR outlier: {(np.sum(outlier_mask_upper)/len(rank))*100:.3f}% Lower IQR outlier: {(np.sum(outlier_mask_lower)/len(rank))*100:.3f}%'
    print('Outlier percentage:', outlier_per)
    return outlier_neuron_no[:,0]

def lower_boundary_threshold(rank, percentage=0.2):
    rank_upper, rank_lower = np.split(rank,[int((1 - (percentage/100)) * rank.shape[0])])
    return rank_lower[:,0]

def order_by_trial(rmat_act, olfactory_path, visual_path, masked_trial_no):
    f_olfactory = h5py.File(olfactory_path, 'r') # olfactory trial sequences
    f_visual = h5py.File(visual_path, 'r') # visual trial sequences

    olfactory = f_olfactory[list(f_olfactory.keys())[-1]]
    ref = [olfactory[0,0], olfactory[1,0]]
    olfactory_vertical = np.array(f_olfactory[ref[0]]).squeeze() # matlab file conversion swap col/row
    olfactory_angled = np.array(f_olfactory[ref[1]]).squeeze()

    visual = f_visual[list(f_visual.keys())[-1]]
    ref = [visual[0,0], visual[1,0]]
    visual_vertical = np.array(f_visual[ref[0]]).squeeze()
    visual_angled = np.array(f_visual[ref[1]]).squeeze()

    stimuli_mark = np.concatenate((np.ones(len(visual_vertical)), np.ones(len(olfactory_vertical))*2, np.ones(len(visual_angled))*3, np.ones(len(olfactory_angled))*4))
    seq_trial = np.concatenate((visual_vertical, olfactory_vertical, visual_angled, olfactory_angled))
    # mask out the nan
    seq_trial = seq_trial[masked_trial_no]
    stimuli_mark = stimuli_mark[masked_trial_no]
    # init index
    index_seq_trial = np.arange(len(seq_trial))
    # Sort by seq w/o nan
    sorted_index_seq_trial = np.array([x for y, x in sorted(zip(seq_trial, index_seq_trial),key=lambda pair: pair[0])])
    sorted_stimuli_mark = np.array([x for y, x in sorted(zip(seq_trial, stimuli_mark), key=lambda pair: pair[0])])
    
    return sorted_index_seq_trial, sorted_stimuli_mark, stimuli_mark

# Loop to find which trial include nan and return the trial no. with nan
def find_nan_trial(rmat_act):
    counter = []
    trial_no = []
    for i in range(rmat_act.shape[2]):
        if np.isnan(np.min(rmat_act[:,:,i])):
            counter.append((i,np.count_nonzero(np.isnan(rmat_act[:,:,i]))))
            trial_no.append(i)
    return counter, trial_no # return the trial no. with nan