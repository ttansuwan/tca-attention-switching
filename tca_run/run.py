import numpy as np
import util
import os
import argparse
import sys
import tensor_decomposition
import cross_val
from multiprocessing import Pool
from tqdm import tqdm

def run(params):
    mouse_no, outlier_method, normalize_method, args = params
    print(mouse_no, outlier_method, normalize_method)
    normalize = ['unnorm', normalize_method]
    tensor_decomposition.set_methods(normalize_method)
    tca_methods_name = tensor_decomposition.methods_name
    mouse_folders = os.listdir(args.data_dir)
    mouse_full_id = list(filter(lambda x: str(mouse_no) in x, mouse_folders))
    if not mouse_full_id:
        sys.exit('Cannot find the mouse number')
    else:
        mouse_full_id = mouse_full_id[0]

    save_dir = tensor_decomposition.create_folder(args.save_data_dir, mouse_full_id, outlier_method, normalize)
    datas, seq_trial, sorted_marking, marking = tensor_decomposition.load_data(args.data_dir, mouse_full_id, outlier_method, normalize)

    if args.isCrossVal:
        for method, data in zip(tca_methods_name, datas):
            cross_val.perform_cross_val(args.no_components, method, args.replicates_no, data, save_dir)
    else:
        ensemble = tensor_decomposition.ensemble_tca(datas, args.no_components, args.replicates_no)
        tensor_decomposition.create_figure(datas, ensemble, args.no_components, args.replicates_no, save_dir, marking, sorted_marking, seq_trial)

def run_multiprocessing(result_params):
    print("\n\nRunning imap multiprocessing")
    pool = Pool(3)
    result_list_tqdm = []
    for result in tqdm(pool.imap(run, result_params), total=len(result_params)):
        result_list_tqdm.append(result)
    print(f'Work process result: {result_list_tqdm}')

def main():
    # Parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--replicates_no', type=int, default=20)
    parser.add_argument('--no_components', type=int, default=20)
    parser.add_argument('--save_data_dir', type=str, default='./result')
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--cross_val', dest='isCrossVal', action='store_true')
    parser.add_argument('--tca', dest='isCrossVal', action='store_false')
    parser.set_defaults(isCrossVal=False, required=True)
    args = parser.parse_args()

    params = []
    outlier_methods = ['default']
    normalize_methods = ['rescale']
    mouse_nos = ['M70_20141106_B1', 'M80_20141108_B1', 'M75_20141107_B1', 'M81_20141108_B1', 'M89_20141115_B1']
    for mouse_no in mouse_nos:
        for outlier_method in outlier_methods:
            for normalize_method in normalize_methods:
                params.append((mouse_no, outlier_method, normalize_method, args))
    run_multiprocessing(params)

if __name__ == "__main__":
    main()