# Quantum k-medians clustering

Implementation of QKmedians from the paper: [2301.10780](https://arxiv.org/abs/2301.10780).

## Algorithm's pseudocode

![pseudo](figures/pseudocode_QKmed.jpeg)

## Distance calculation quantum circuit

![Distance circuit](figures/DistCirc.png)

## How to run an example?

### Download dataset
Dataset's dimensionality is reduced by passing it through autoencoder. If you are interested more, please refer to [[\*]](https://arxiv.org/abs/2301.10780).\
Reduced dataset can be downloaded from `Zenodo` : 
[record/7673769](https://zenodo.org/record/7673769)\
Small portion of dataset in `data` folder:
  - `latentrep_QCD_sig.h5` $\rightarrow$ train dataset (QCD)
  - `latentrep_QCD_sig_testclustering.h5` $\rightarrow$ test dataset (QCD)
  - `latentrep_RSGraviton_WW_NA_35.h5` $\rightarrow$ test dataset (Signal)

### Run training
To run a training of quantum k-medians algorithm we need to provide arguments:
- `train_size` (int): number of samples for training
- `read_file` (str): path to the training dataset
- `seed` (int): seed for consistent results in training
- `k` (int): number of clusters (`default = 2`)
- `tolerance` (float): convergence tolerance (`default = 1.0e-3`)
- `min_type` (str): minimization type for distance to cluster search (`default = 'classic'`)
- `nshots` (int): number of shots for executing quantum circuit (`default = 10000`)
- `save_dir` (str): path to save results

```python
python train_qkmedians.py --train_size 600 --read_file 'data/latentrep_QCD_sig.h5' --k 2 --seed 123 --tolerance 1e-3 --min_type 'classic' --save_dir 'save_directory'
```

### Run evaluation
To run an evaluation of quantum k-medians algorithm we need to provide arguments:
- `centroids_file` (str): name of the file for saved centroids coordinates
- `data_qcd_file` (str): name of the file for test QCD dataset
- `data_signal_file` (str): name of the file for test signal dataset
- `k` (int): number of clusters (`default = 2`)
- `test_size` (int): number of test samples (`default = 10000`)
- `title` (str): Title of ROC curve plot (`default = 'Anomaly detection results'`)
- `results_dir` (str): path to file with saved centroids
- `data_dir` (str): path to file with test datasets
- `save_dir_roc` (str): path to directory for saving ROC plot
- `xlabel` (str): name of x-axis in ROC plot
- `ylabel` (str): name of y-axis in ROC plot

```python
python evaluate.py --centroids_file 'centroids.npy' --data_qcd_file 'latentrep_QCD_sig_testclustering.h5' --data_signal_file 'latentrep_RSGraviton_WW_NA_35.h5' --results_dir '~' --data_dir 'data' --save_dir_roc '~'
```
