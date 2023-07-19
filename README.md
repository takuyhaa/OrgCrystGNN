# OrgCrystGNN
This repository provides codes for OrgCrytGNN.  
It is possible to compare prediction accuracy using molecular and crystal graphs.  
The paper using OrgCrystGNN has been uploaded on ChemRxiv (https://doi.org/10.26434/chemrxiv-2022-q426t-v2).
  
![image](https://github.com/takuyhaa/OrgCrystGNN/assets/86113952/c6978bb3-a3b9-4814-81f9-4d8450bcc304)  

## Dependencies
We implemented the codes on Python 3.7 in a computer of Windows 10.  
Minimum packages we used are following.  
- torch == 1.12.1+cu116
- torch-cluster == 1.6.0
- torch-geometric == 2.1.0.post1
- torch-scatter == 2.0.9
- torch-sparse == 0.6.15
- torch-spline-conv == 1.2.1 
- torchaudio == 0.12.1+cu116  
- torchvision == 0.13.1+cu116 
- pickle == 4.0
- numpy == 1.23.3
- optuna == 3.0.2
- scikit-learn == 1.1.2
- pandas == 1.4.4
- rdkit == 2022.03.5
- scipy == 1.8.1
- ase == 3.22.1

## Dataset
The original dataset of band gap can be downloaded from organic materials database, OMDB (https://omdb.mathub.io/).  
The curated dataset can be obtained by following the instruction of notebooks/0_dataset-preparation.ipynb.  
Other datasets can be utilized for model training and inference.  
In that case, please prepare for dataset in csv format.  
  
Ex.) Dataset structure  
<img width="324" alt="image" src="https://github.com/takuyhaa/OrgCrystGNN/assets/86113952/8c5951f1-51e2-4a46-bc90-559ab7ecdc4f">

## Files
Each executable file has the following roles.  
`main.py`: Work mainly  
`train.py`: Train the machine learning models  
`model.py`: Define the GNN models  
`process.py`: Calculate molecular descriptors  
`settings.py`: Input the setting  

## Setting arguments
--- Specipy work details
- work_name = 'Large-screening_TUWGEX-' # Work name for save
- run_mode = 'Training' # {Training, Predict, Explain, Hyperopt}
- data_mode = 'Crystal' # {Crystal, Molecule}
- model_name = 'MEGNet' # {MEGNet, GCN, CGCNN, SchNet}
- train_size = 6000 #8457
- fingerprint = 'Avalon'
- random_state = 2
- model_path = None
- load_model = None
- cuda_id = 'cuda:0'

--- Specify working directory
- root = 'D:datasets/Bandgap_rev/'
- root_train = 'trainset/'
- root_val = 'valset/'
- root_test = 'testset/'
- file_name_raw = 'smiles-cod-bandgap.csv'
- file_name_train = 'dataset_train.csv'
- file_name_val = 'dataset_val.csv'
- file_name_test = 'dataset_test.csv'

--- Specify graph parameters
- graph_max_radius = 8
- graph_max_neighbors = 12
- addH = False

--- Initial data split
- train_ratio = 0.8
- val_ratio = 0.05
- test_ratio = 0.15
- rand_split = 1
- split_data = False

--- Training
- batch_size = 64 # No change
- n_epoch = 200   # No change
- lr = 0.001

--- GNN Model
- dim1 = 90
- dim2 = 70
- dim3 = 210
- gc_count = 3
- post_fc_count = 2
- pool = "global_max_pool"

Other arguments are available, but I did not change them in original article.  

## Example
To run:  
`python main.py`
