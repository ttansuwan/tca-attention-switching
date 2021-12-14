# Applying Tensor Decomposition Methods to Neural Population Recordings

The goal of this project is to apply tensor decomposition methods to data recorded from populations of neurons in visual cortex of mice as they switch between a visual discrimination task and an olfactory discrimination task. We mainly observed the structure identified by the TCA method and its implications for how task-switching influence neuronal population activity in visual cortex.

## Installation

Install the dependencies of this project via requirements.txt.

```bash
pip install -r requirements.txt
```

## Usage

Preprocess data tensor
```bash
python3 preprocess/run.py 
```
Run TCA
```bash
python3 tca_run/run.py --replicates_no {int} --no_components {int} --save_data_dir {str} --data_dir {str} --tca 
```
Run TCA cross validation
```bash
python3 tca_run/run.py --replicates_no {int} --no_components {int} --save_data_dir {str} --data_dir {str} --cross_val
``` 
Refitting 
```bash
python3 tca_run/refitting.py --processed_dir {str}--data_dir {str} --mouse_no {int}
``` 

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## Acknowlegments
Thank you Angus Chadwick for supporting me throughout my thesis process and UoE for giving me an opportunity to study there for my master.

This project mainly used [tensortools](https://github.com/ahwillia/tensortools), and hypothesize based on this [paper](https://doi.org/10.1016/j.neuron.2018.05.015). 

## Resources
My slides are available [here.](https://docs.google.com/presentation/d/1PK00tYdbBBXcCSz8m6EeeMM9DGtYjYlOCHVbJd2NRW4/edit?usp=sharing) I sometimes update the slides so please check for any updates and the thesis is available in this github.

The data tested on this project is not publicly available.

## License
[MIT](https://choosealicense.com/licenses/mit/)