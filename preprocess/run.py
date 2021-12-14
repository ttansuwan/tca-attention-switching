from glob import glob
from preprocess_data import run

list_of_mouse = glob('./matlab_result/*/')

percentages = [5, 10, 15, 20, 30]

with open('./processed/mice_dimensions.csv', mode='w') as csv_file:
    csv_file.write('mouse_no,neuron_no,temporal_no,trial_no\n')

for mouse_no in list_of_mouse:
    mouse_no = mouse_no.split('/')[2]
    for per in percentages:
        run(mouse_no, 'snr', per)
    run(mouse_no)