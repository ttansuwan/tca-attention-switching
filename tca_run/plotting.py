import numpy as np
import matplotlib.pyplot as plt
import glob
import tensortools as tt
import util
import subprocess
import argparse
from multiprocessing import Pool
import os
from tqdm import tqdm
import timeit

def plot(args):
    U, method_name, components, replicate, save_data_dir, marking, seq_trial = args
    try:
        util.plotting_factors(U, method_name, components, replicate, save_data_dir, marking, seq_trial)
        message =  f"{os.getpid()}: success"
    except Exception as e:
        message =  f"{os.getpid()}: error -> {e}" 
    return message

def run_plot_multiprocessing(result_params):
    tic=timeit.default_timer()
    print("\n\nRunning imap multiprocessing")
    pool =  Pool(1)
    result_list_tqdm = []
    for result in tqdm(pool.imap(plot, result_params), total=len(result_params)):
        result_list_tqdm.append(result)
    toc=timeit.default_timer()
    print(f'Done={toc - tic}')

def main():
    # Parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--processed_dir', type=str, required=True)
    parser.add_argument('--mouse_no', type=str, required=True)
    args = parser.parse_args()

    result_params = []
    mouse_no = args.mouse_no

    path = f'{args.data_dir}/{mouse_no}'
    list_npy = glob.glob(f'{path}/**/*.npy', recursive=True)
    marking = np.load(f'./{args.processed_dir}/{mouse_no}/stimuli_mark.npy')
    sorted_marking = np.load(f'./{args.processed_dir}/{mouse_no}/sorted_stimuli_mark.npy')
    seq_trial = np.load(f'./{args.processed_dir}/{mouse_no}/index_seq_trial.npy')

    print(list_npy)
    for i in list_npy:
        data = np.load(i, allow_pickle=True)

        U = tt.KTensor(data)
        i = i[:-4] # remove .npy
        path_split = i.split('/')
        save_dir = '/'.join(path_split[:5])

        filename_split = path_split[-1].split('-')
        setting = [] # method, components, replicate
        for j in filename_split:
            setting.append(j.split('=')[1])
        result_params.append((U, setting[0], int(setting[1])-1, int(setting[2]), save_dir, marking, None))
        result_params.append([U, setting[0], int(setting[1])-1, int(setting[2]), save_dir, sorted_marking, seq_trial])

    run_plot_multiprocessing(result_params=result_params)
if __name__ == "__main__":
    main()