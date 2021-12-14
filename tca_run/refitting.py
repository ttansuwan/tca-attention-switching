import numpy as np
import matplotlib.pyplot as plt
import glob
import tensortools as tt
import util
import argparse
from multiprocessing import Pool
import os
from tqdm import tqdm
import csv
from collections import defaultdict

def write_csv(cell_label, index, mouse_no, component_no, percentage, s_, method_name, cell_info, save_data_dir):
    csv_file = f'{save_data_dir}/{method_name}/refitting/cell_label_count.csv'
    
    dict_data = {'mouse_name':mouse_no, 'method_name':method_name,'component_no':component_no+1, 'percentage':percentage, 's_v|s_o':s_, 
                    'cell_1':cell_label.get(1), 'cell_2':cell_label.get(2), 'cell_3':cell_label.get(3), 'cell_4':cell_label.get(4), 'cell_5':cell_label.get(5), 
                    'keep_cell': cell_info[0], 'remove_cell': cell_info[1], 'index':index[:10]}
    try:
        with open(csv_file, 'a') as csvfile:
            csv_columns = ['mouse_name', 'method_name','component_no', 'percentage', 's_v|s_o', 'cell_1', 'cell_2', 'cell_3', 'cell_4', 'cell_5', 'keep_cell', 'remove_cell', 'index']
            writer = csv.DictWriter(csvfile, csv_columns)
            writer.writerow(dict_data)
    except IOError:
        print("I/O error")

def refitting(run_factors, cell_label, org_data, marking):
    list_stimuli = util.angled_vertical_diff(run_factors[2], marking)

    for i in range(len(list_stimuli)):
        temp_ = (i,)
        list_stimuli[i] = list_stimuli[i] + temp_

    sort_stimuli_diff = sorted(list_stimuli, key=lambda pair: pair[0:2], reverse=True)
    max_stimuli_diff = [sort_stimuli_diff[0][0], sort_stimuli_diff[0][1]]
    neuron_factor = run_factors[0]

    # Identifying which neuron(s) activity are high in the component with high stimuli diff
    factor_no = sort_stimuli_diff[0][2] # Get the component no of the highest stimuli diff
    slice_U = np.absolute(neuron_factor[:,factor_no+1]) # Reminder = neuron factor did not get sorted previously
    sort_index = np.argsort(slice_U, axis=0)[::-1][:slice_U.shape[0]] # Sort neuron by its magnitude

    save_U = []
    for per in range(5):
        cell_info = []
        # x % of the U removed 
        percentage = (per+1)*0.1

        remove_neuron_index, keep_neuron_index = np.split(sort_index,[int(percentage * len(sort_index))])
        cell_info.extend([keep_neuron_index.shape[0], remove_neuron_index.shape[0]])

        temp_org_data = np.delete(org_data, remove_neuron_index, 0) # delete marked cell from org data

        unique, counts =  np.unique(cell_label, return_counts=True)
        cell_unique = dict(zip(unique, counts))

        cell_label_remove = cell_label[remove_neuron_index] # cell labels that are removed
        unique, counts = np.unique(cell_label_remove, return_counts=True)
        cell_unique_remove = dict(zip(unique, counts))

        for i in range(5):
            y = cell_unique.get(i+1)
            x = cell_unique_remove.get(i+1)
            if y:
                if x:
                    cell_unique_remove[i+1] = (x/y, x) # calculate the percentage of removed neuron
        save_U.append([temp_org_data, percentage, cell_unique_remove, remove_neuron_index, cell_info])
    return save_U

def plot(args):
    run_factors, mouse_no, method_name, component_no, replicate, save_data_dir, marking, sorted_marking, seq_trial, cell_label, org_data = args
    # try:
    U_per_list = refitting(run_factors, cell_label, org_data, marking)
    for U in U_per_list:
        U[1] = int(U[1] * 100)
        if method_name == 'ncp_als':
            U_fit= tt.ncp_hals(U[0], rank=component_no, verbose=False)
        else:
            U_fit = tt.cp_als(U[0], rank=component_no+1, verbose=False)
        diff_stimuli = np.array(util.angled_vertical_diff(U_fit.factors[2], marking))
        max_diff = np.max(diff_stimuli, axis=0)
        write_csv(U[2], U[3], mouse_no, component_no, U[1], max_diff, method_name, U[4], save_data_dir)

        util.plotting_factors(U_fit.factors, method_name, component_no, replicate, save_data_dir, marking, None, refitting_per=U[1])
        util.plotting_factors(U_fit.factors, method_name, component_no, replicate, save_data_dir, sorted_marking, seq_trial, refitting_per=U[1])
    message =  f"{os.getpid()}: success"
    # except Exception as e:
        # message =  f"{os.getpid()}: error -> {e}" 
    return message

def run_plot_multiprocessing(result_params):
    print("\n\nRunning imap multiprocessing")
    pool = Pool(3)
    result_list_tqdm = []
    for result in tqdm(pool.imap(plot, result_params), total=len(result_params)):
        result_list_tqdm.append(result)
    print(f'Work process result: {result_list_tqdm}')

def main():
    # Parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--processed_dir', type=str, required=True)
    parser.add_argument('--mouse_no', type=str, required=True)
    args = parser.parse_args()

    result_params = []
    x = [args.mouse_no]
    for mouse_no in x:
        path = f'{args.data_dir}/{mouse_no}'
        list_npy = glob.glob(f'{path}/**/*.npy', recursive=True)
        marking = np.load(f'{args.processed_dir}/{mouse_no}/stimuli_mark.npy')
        sorted_marking = np.load(f'{args.processed_dir}/{mouse_no}/sorted_stimuli_mark.npy')
        seq_trial = np.load(f'{args.processed_dir}/{mouse_no}/index_seq_trial.npy')
        cell_label = np.load(f'{args.processed_dir}/{mouse_no}/result/snr-5/cell_label.npy')
        org_data = np.load(f'./{args.processed_dir}/{mouse_no}/result/snr-5/processed_unnorm.npy')

        if not os.path.exists(f'{path}/snr-5/unnorm-unnorm/cp_als/refitting'):
            os.makedirs(f'{path}/snr-5/unnorm-unnorm/cp_als/refitting/order')
            os.makedirs(f'{path}/snr-5/unnorm-unnorm/cp_als/refitting/no_order')
            for i in range(5):
                percentage = int((i+1)*10)
                os.makedirs(f'{path}/snr-5/unnorm-unnorm/cp_als/refitting/order/{percentage}_remove')
                os.makedirs(f'{path}/snr-5/unnorm-unnorm/cp_als/refitting/no_order/{percentage}_remove')

        # Writing header
        csv_file = f'{path}/snr-5/unnorm-unnorm/cp_als/refitting/cell_label_count.csv'
        try:
            with open(csv_file, 'w') as csvfile:
                csv_columns = ['mouse_name', 'method_name','component_no', 'percentage', 's_v|s_o', 'cell_1', 'cell_2', 'cell_3', 'cell_4', 'cell_5', 'keep_cell', 'remove_cell', 'index']
                writer = csv.writer(csvfile)
                writer.writerow(csv_columns)
        except IOError:
            print('Cannot write header', IOError)
        
        for i in list_npy:
            path_split = i[:-4].split('/')
            save_dir = '/'.join(path_split[:5])
            filename_split = path_split[-1].split('-')
            if int(filename_split[1].split('=')[1]) < 8:
                pass
            else:
                data = np.load(i, allow_pickle=True)
                setting = [] # method, components, replicate
                for j in filename_split:
                    setting.append(j.split('=')[1])
                result_params.append([data, mouse_no, setting[0], int(setting[1])-1, int(setting[2]), save_dir, marking, sorted_marking, seq_trial, cell_label, org_data])
    run_plot_multiprocessing(result_params=result_params)

if __name__ == "__main__":
    main()