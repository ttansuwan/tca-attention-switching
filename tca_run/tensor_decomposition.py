import numpy as np
import tensortools as tt
import matplotlib.pyplot as plt
import util
import os
import argparse
import sys
from copy import deepcopy

methods = [
  'cp_als',    # fits unconstrained tensor decomposition.
  'ncp_hals'  # fits nonnegative tensor decomposition.
]

methods_name = ["", ""]

plot_options = {
  'cp_als': {
    'line_kw': {
      'color': 'red',
      'label': 'cp_als',
    },
    'scatter_kw': {
      'color': 'red',
    },
  },
  'ncp_hals': {
    'line_kw': {
      'color': 'blue',
      'alpha': 0.5,
      'label': 'ncp_hals',
    },
    'scatter_kw': {
      'color': 'blue',
      'alpha': 0.5,
    }
  }
}

def set_methods(normalize_method):
  global methods, methods_name, plot_options
  if normalize_method == 'zscore':
    methods[1] = 'cp_als'
    methods_name[0] = f'{methods[0]}_0'
    methods_name[1] = f'{methods[1]}_1'
    plot_options[methods_name[0]] = {
      'line_kw': {
        'color': 'red',
        'label': 'cp_als_0',
      },
      'scatter_kw': {
        'color': 'red',
      },
    }
    plot_options[methods_name[1]] = {
      'line_kw': {
        'color': 'blue',
        'alpha': 0.5,
        'label': 'cp_als_1',
      },
      'scatter_kw': {
        'color': 'blue',
        'alpha': 0.5,
      }
    }
  else:
    methods[1] = 'ncp_hals'
    methods_name = deepcopy(methods)

def create_folder(save_data_dir, mouse_full_id, outlier_method, normalize_methods):
  global methods, methods_name
  # Setting save folders
  save_dir = f'{save_data_dir}/{mouse_full_id}'
  if outlier_method is not None:
    save_dir = f'{save_dir}/{outlier_method}'
  else:
    save_dir = f'{save_dir}/default'

  # Create folder the normalize method
  save_dir = f'{save_dir}/{normalize_methods[0]}-{normalize_methods[1]}'

  if not os.path.exists(save_dir):
      os.makedirs(save_dir)
  
  for i in methods_name:
    temp_dir = f'{save_dir}/{i}'
    if not os.path.exists(temp_dir):
      os.makedirs(f'{temp_dir}/order')
      os.makedirs(f'{temp_dir}/no_order')
  return save_dir

def load_data(data_dir, mouse_full_id, outlier_method, normalize_methods):
  # Data loading - processed npy
  data = []
  for norm_method in normalize_methods:
    data_path = data_dir + "/" + mouse_full_id
    data_path = data_path + '/result'

    if outlier_method is not None:
      data_path = data_path + "/" + outlier_method
    else:
      data_path = data_path + "/default"
    
    # Find the normalize method .npy in the folder
    norm_files = [f for f in os.listdir(data_path) if f.endswith('.npy')]
    norm_file = list(filter(lambda x: str(norm_method) in x, norm_files))
    if not norm_file:
      sys.exit('Cannot find the normalize method')
    else:
      norm_files = norm_file[0]
    data_path = data_path + "/" + norm_files
    data.append(np.load(data_path))
  # Data loading - seq trial/stimuli mark npy
  seq_trial = np.load(f'{data_dir}/{mouse_full_id}/index_seq_trial.npy')
  sorted_marking = np.load(f'{data_dir}/{mouse_full_id}/sorted_stimuli_mark.npy')
  marking = np.load(f'{data_dir}/{mouse_full_id}/stimuli_mark.npy')
  return data, seq_trial, sorted_marking, marking

def ensemble_tca(data, num_components, replicates_no):
  global methods, plot_options

  ensembles = []
  for i in range(len(methods)):
    ensembles.append(tt.Ensemble(fit_method=methods[i]))
    ensembles[-1].fit(data[i], ranks=range(1, num_components+1), replicates=replicates_no)
  return ensembles

def create_figure(datas, ensembles, num_components, replicates_no, save_dir, marking, sorted_marking, seq_trial):
  global methods

  # Plot similarity and error plots.
  plt.figure()
  for m, m_name in zip(ensembles, methods_name):
    x = plot_options[m_name]
    tt.plot_objective(m, **plot_options[m_name])
  plt.legend()
  plt.savefig(f'{save_dir}/obj_plot.png')
  plt.clf()

  plt.figure()
  for m, m_name in zip(ensembles, methods_name):
      tt.plot_similarity(m, **plot_options[m_name])
  plt.legend()
  plt.savefig(f'{save_dir}/sim_plot.png')
  plt.clf()
  plt.close()

  # Plot the low-d factors for an example model, e.g. rank-2, first optimization run / replicate.
  replicate = 0 # For plotting factors
  for method, data, method_name, ensemble in zip(methods, datas, methods_name, ensembles):
    R_2 = []
    for components in range(num_components):
      U = ensemble.factors(components+1)[replicate]
      # Save factors before sorting
      repack = np.array([U[0], U[1], U[2]], dtype="object") # remove warning 
      np.save(f'{save_dir}/{method_name}/method={method_name}-factors_num_components={components+1}-replicate={replicate}.npy', repack, allow_pickle=True)
      
      temp_ = []
      for r in range(replicates_no):
          U = ensemble.factors(components+1)[r].full()
          result = util.r_squared(U, data)
          temp_.append(result)
      R_2.append(temp_)
    util.plot_r_squared(R_2, method_name, num_components, save_dir)