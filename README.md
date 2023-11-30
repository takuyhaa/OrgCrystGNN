# OrgCrystGNN
This repository provides codes for OrgCrytGNN.  
It is possible to compare prediction accuracy using molecular and crystal graphs.  
When you use or refer this repository, please cite the published paper (https://doi.org/10.1021/acsomega.3c05224).
  
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
- `work_name` (str): Work name for save
- `run_mode` (str): {Training, Predict, Hyperopt}
- `data_mode` (str): {Crystal, Molecule}
- `model_name` (str): {MEGNet, GCN, CGCNN, SchNet}
- `train_size` (int): train size
- `random_state` (int): random state for data split
- `model_path` (int): Parent folder of trained model for prediction mode
- `load_model` (int): Trained model name for prediction mode
- `cuda_id` (int): cuda identifier (ex. 'cuda:0')

--- Specify working directory
- `root` (str): Parent folder of working
- `root_train` (str): Train folder in the parent folder
- `root_val` (str): Validation folder in the parent folder
- `root_test` (str): Test folder in the parent folder
- `file_name_raw` (str): File name of the dataset before train-val-test split stored in the parent folder (ex. 'smiles-cod-bandgap.csv')
- `file_name_train` (str): File name of training dataset after split
- `file_name_val` (str): File name of validation dataset after split
- `file_name_test` (str): File name of test dataset after split

--- Specify graph parameters
- `graph_max_radius` (float): max radius for edge formation of crystal graph
- `graph_max_neighbors` (int): max edges of a node in crystal graph
- `addH` (bool): Wether hydrogen atoms are added in molecular graph {True, False}

--- Initial data split
- `train_ratio` = 0.8
- `val_ratio` = 0.05
- `test_ratio` = 0.15
- `rand_split` = 1
- `split_data` = False

--- Training
- `batch_size` = 64 # No change
- `n_epoch` = 200   # No change
- `lr` = 0.001

--- GNN Model
- `dim1` = 90
- `dim2` = 70
- `dim3` = 210
- `gc_count` = 3
- `post_fc_count` = 2
- `pool` = "global_max_pool"

Other arguments are available, but I did not change them in original article.  

## Example
To run:  
`python main.py`

Ex. Folder structure after initial data execution
![image](https://github.com/takuyhaa/OrgCrystGNN/assets/86113952/51ad06eb-e5a1-44b5-9e2a-2977cd7cc95c)
