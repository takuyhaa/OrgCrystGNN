## Import packages
import numpy as np
import pandas as pd
from rdkit import Chem
import torch
from torch_geometric.data import Dataset, Data
from torch.utils.data import TensorDataset
from torch.nn.functional import one_hot
import os
from ase import io
from scipy.stats import rankdata
from torch_geometric.utils import dense_to_sparse, degree, add_self_loops
import warnings
from rdkit.Chem import AllChem, DataStructs, Descriptors, ReducedGraphs, MACCSkeys
from rdkit.Avalon.pyAvalonTools import GetAvalonFP
from rdkit.Chem.EState import Fingerprinter
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
warnings.simplefilter('ignore')


###############################################################################
class MoleculeDataset(Dataset):
    def __init__(self, root, filename, datasize=None, test=False,transform=None, pre_transform=None, addH=False, random_state=0):
        """
        root = Where the dataset should be stored. This folder is split
        into raw_dir (downloaded dataset) and processed_dir (processed data). 
        """
        
        self.filename = filename
        self.addH = addH
        self.datasize = datasize
        self.test = test
        self.random_state = random_state
        super(MoleculeDataset, self).__init__(root, transform, pre_transform)
        
    @property
    def raw_file_names(self):
        """ If this file exists in raw_dir, the download is not triggered.
            (The download func. is not implemented here)  
        """
        return self.filename

    @property
    def processed_file_names(self):
        """ If these files are found in raw_dir, processing is skipped"""
        self.data = pd.read_csv(self.raw_paths[0]).reset_index()

        if self.test:
            return [f'data_test_{i}.pt' for i in list(self.data.index)]
        else:
            return [f'data_{i}.pt' for i in list(self.data.index)]

    def download(self):
        pass
    
    def process(self):
        self.data = pd.read_csv(self.raw_paths[0])
        if self.datasize == None:
            print(self.data.shape)
        else: 
            if self.data.shape[0] > self.datasize:
                self.data = self.data.sample(
                    n=self.datasize,
                    random_state=self.random_state
                ).reset_index(drop=True)
                print('Data size:', self.data.shape[0])
        if 'data_0.pt' not in os.listdir(self.processed_dir):
            for index, item in self.data.iterrows():
                mol_obj = Chem.MolFromSmiles(item[0])
                if self.addH is True:
                    mol_obj = Chem.AddHs(mol_obj)
                # Get node features
                node_feats = self._get_node_features(mol_obj)
                # Get edge features
                edge_feats = self._get_edge_features(mol_obj)
                # Get adjacency info
                edge_index = self._get_adjacency_info(mol_obj)
                # Get labels info
                label = self._get_labels(item[2])
                # Make glogal values
                try:
                    u = self._get_labels(item[3])
                except:
                    u = np.zeros((3))
                    u = torch.Tensor(u[np.newaxis, ...])
                # Make edge weight
                edge_weight = np.ones((edge_feats.shape[0]))
                edge_weight = torch.tensor(edge_weight, dtype=torch.float)
                # Make ID 
                structure_id = [[item[0]]]
                # # Get fingerprint
                # fingerprint = self._get_fingerprint(mol_obj)

                # Create data object
                data = Data(x=node_feats, 
                            edge_index=edge_index,
                            edge_attr=edge_feats,
                            edge_weight=edge_weight,
                            y=label,
                            structure_id=structure_id,
                            u=u,
                            ) 
                torch.save(data, os.path.join(self.processed_dir, f'data_{index}.pt'))

    def _get_node_features(self, mol):
        """ 
        This will return a matrix / 2d array of the shape
        [Number of Nodes, Node Feature size]
        """
        all_node_feats = []

        for atom in mol.GetAtoms():
            node_feats = []
            # Feature 1: Atomic number
            node_feats = one_hot(
                torch.tensor(atom.GetAtomicNum()-1), num_classes=113
            )
            node_feats = node_feats.tolist()       
            # Feature 2: Atom degree
            node_feats.append(atom.GetDegree())
            # Feature 3: Formal charge
            node_feats.append(atom.GetFormalCharge())
            # Feature 4: Hybridization
            node_feats.append(atom.GetHybridization())
            # Feature 5: Aromaticity
            node_feats.append(atom.GetIsAromatic())
            # Feature 6: Total Num Hs
            node_feats.append(atom.GetTotalNumHs())
            # Feature 7: Radical Electrons
            node_feats.append(atom.GetNumRadicalElectrons())
            # Feature 8: In Ring
            node_feats.append(atom.IsInRing())
            # Feature 9: Chirality
            node_feats.append(atom.GetChiralTag())

            # Append node features to matrix
            all_node_feats.append(node_feats)

        all_node_feats = np.asarray(all_node_feats)
        return torch.tensor(all_node_feats, dtype=torch.float)

    def _get_edge_features(self, mol):
        all_edge_feats = []

        for bond in mol.GetBonds():
            edge_feats = []
            # Feature 1: Bond type (as double)
            edge_feats.append(torch.tensor(bond.GetBondTypeAsDouble()))
            # Feature 2: Rings
            edge_feats.append(bond.IsInRing())
            # Append node features to matrix (twice, per direction)
            all_edge_feats += [edge_feats, edge_feats]

        self_len = len(mol.GetAtoms())
        for i in range(self_len):
            all_edge_feats += [[torch.tensor(0), False]]

        all_edge_feats = np.asarray(all_edge_feats)
        return torch.tensor(all_edge_feats, dtype=torch.float)

    def _get_adjacency_info(self, mol):
        edge_indices = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_indices += [[i, j], [j, i]]

        edge_indices = torch.tensor(edge_indices).view(2, -1)
        self_loops = True
        if self_loops == True:
            edge_indices, edge_weight = add_self_loops(
                edge_indices, num_nodes=len(mol.GetAtoms())
            )
        edge_indices = edge_indices.to(torch.long)#.view(2, -1)
        return edge_indices

    def _get_labels(self, label):
        label = np.asarray([label])
        label = torch.tensor(label, dtype=torch.float32)
        return torch.reshape(label, (1,1))

    def len(self):
        return self.data.shape[0]

    def get(self, idx):
        if self.test:
            data = torch.load(os.path.join(self.processed_dir, 
                                 f'data_test_{idx}.pt'))
        else:
            data = torch.load(os.path.join(self.processed_dir, 
                                 f'data_{idx}.pt'))   
        return data
    

###############################################################################
class CrystalDataset(Dataset):
    def __init__(self, root, filename, r_max, n_neighbors, 
                 datasize=None, test=False, transform=None, 
                 pre_transform=None, random_state=None):
        """
        root = Where the dataset should be stored. This folder is split
        into raw_dir (downloaded dataset) and processed_dir (processed data). 
        """
        self.filename = filename
        self.datasize = datasize
        self.test = test
        self.random_state = random_state
        self.r_max = r_max
        self.n_neighbors = n_neighbors
        super(CrystalDataset, self).__init__(root, transform, pre_transform)
        
    @property
    def raw_file_names(self):
        """ If this file exists in raw_dir, the download is not triggered.
            (The download func. is not implemented here)  
        """
        return self.filename

    @property
    def processed_file_names(self):
        """ If these files are found in raw_dir, processing is skipped"""
        self.data = pd.read_csv(self.raw_paths[0]).reset_index()

        if self.test:
            return [f'data_test_{i}.pt' for i in list(self.data.index)]
        else:
            return [f'data_{i}.pt' for i in list(self.data.index)]

    def download(self):
        pass

    def process(self):
        self.data = pd.read_csv(self.raw_paths[0])
        if self.datasize == None:
            print(self.data.shape)
        else: 
            if self.data.shape[0] > self.datasize:
                self.data = self.data.sample(
                    n=self.datasize,
                    random_state=self.random_state
                ).reset_index(drop=True)
                print('Data size:', self.data.shape[0])
        if 'data_0.pt' not in os.listdir(self.processed_dir):
            for index, item in self.data.iterrows():
                if type(item[1]) is str:
                    ase_crystal = io.read(self.root + '/../NNP_modulus_raw/' + item[1] + '.cif')
                else:
                    ase_crystal = io.read(self.root + '/../NNP_modulus_raw/' + str(int(item[1])) + '.cif')
                # Get node features
                node_feats = self._get_node_features(ase_crystal)
                # Get adjacency info & weight
                edge_index, edge_weight = self._get_adjacency_info(ase_crystal)
                # Get edge features
                edge_attr = self._get_edge_features(edge_weight)
                # Get labels info
                label = self._get_labels(item[2])
                # Make glogal values
                try:
                    u = self._get_labels(item[3])
                except:
                    u = np.zeros((3))
                    u = torch.Tensor(u[np.newaxis, ...])
                # Make ID 
                structure_id = [[item[1]]]

                # Create data object
                data = Data(x=node_feats, 
                            edge_index=edge_index,
                            edge_attr=edge_attr,
                            edge_weight=edge_weight,
                            y=label,
                            structure_id=structure_id,
                            u=u
                            ) 
                torch.save(data, os.path.join(self.processed_dir, f'data_{index}.pt'))
    
    
    ##Selects edges with distance threshold and limited number of neighbors
    def _trimming(self, matrix, r_max, n_neighbors):
        mask = matrix > r_max
        matrix_trimmed = np.ma.array(matrix, mask=mask)
        matrix_trimmed = rankdata(matrix_trimmed, method="ordinal", axis=1)
        matrix_trimmed = np.nan_to_num(np.where(mask, np.nan, matrix_trimmed))
        matrix_trimmed[matrix_trimmed > n_neighbors + 1] = 0
        matrix_trimmed = np.where(matrix_trimmed == 0, matrix_trimmed, matrix)
        return matrix_trimmed
                
    def NormalizeEdge(self, edge_weight):
        # edge_weight_norm = (edge_weight - edge_weight.min()) /(edge_weight.max() - edge_weight.min())
        edge_weight_norm = edge_weight
        return edge_weight_norm
    
    def distance_gaussian(self, edge_weight, start=0.0, stop=5.0, resolution=50, coef=0.5):
        offset = torch.linspace(start, stop, resolution)
        edge_weight = edge_weight.unsqueeze(-1) - offset.view(1, -1)
        return torch.exp(-1*torch.pow(edge_weight, 2)/coef**2)

    def _get_node_features(self, ase_crystal):
        all_node = torch.tensor(ase_crystal.get_atomic_numbers()).to(torch.int64)
        all_node_feats = one_hot(all_node, 113)
        return all_node_feats.to(torch.float)

                
    def _get_edge_features(self, edge_weight):
        edge_attr = self.distance_gaussian(edge_weight)
        return edge_attr.to(torch.float)

    def _get_adjacency_info(self, ase_crystal):
        ##Obtain distance matrix with ase
        distance_matrix = ase_crystal.get_all_distances(mic=True)

        ##Create sparse graph from distance matrix
        distance_matrix_trimmed = self._trimming(
            matrix=distance_matrix,
            r_max=self.r_max,
            n_neighbors=self.n_neighbors)
        distance_matrix_trimmed = torch.Tensor(distance_matrix_trimmed)
        out = dense_to_sparse(distance_matrix_trimmed)
        edge_index = out[0]
        edge_weight = out[1]

        self_loops = True
        if self_loops == True:
            edge_index, edge_weight = add_self_loops(
                edge_index, edge_weight, num_nodes=len(ase_crystal), fill_value=0
            )
        return edge_index, edge_weight

                
    def _get_labels(self, label):
        label = np.asarray([label])
        label = torch.tensor(label, dtype=torch.float32)
        return torch.reshape(label, (1,1))

                
    def len(self):
        return self.data.shape[0]

                
    def get(self, idx):
        """ - Equivalent to __getitem__ in pytorch
            - Is not needed for PyG's InMemoryDataset
        """
        if self.test:
            data = torch.load(os.path.join(self.processed_dir, 
                                 f'data_test_{idx}.pt'))
        else:
            data = torch.load(os.path.join(self.processed_dir, 
                                 f'data_{idx}.pt'))   
        return data


#################################################3
class MoleculeFPDataset:
    def __init__(self, root, filename, datasize=None, fingerprint=None, random_state=0):
        self.root = root
        self.filename = filename
        self.datasize = datasize
        self.fingerprint = fingerprint
        self.random_state = random_state

    def mol2vec(self, smiles, fptype='ECFP', radius=2, bits = 1024):
        vector = np.zeros((1,))
        mol = Chem.MolFromSmiles(smiles)
        if fptype == 'ECFP':
            DataStructs.ConvertToNumpyArray(
                AllChem.GetMorganFingerprintAsBitVect(mol, radius, bits), vector
            )
        elif fptype == 'Avalon':
            DataStructs.ConvertToNumpyArray(
                GetAvalonFP(mol), vector
            )
        elif fptype == 'MACCSKeys':
            DataStructs.ConvertToNumpyArray(
                AllChem.GetMACCSKeysFingerprint(mol), vector
            )
        elif fptype == 'ErG':
            vector = ReducedGraphs.GetErGFingerprint(mol)
        elif fptype == 'Estate':
            vector = Fingerprinter.FingerprintMol(mol)[0]
        else:
            raise TypeError()
        return vector
        
    def file2data(self):
        df = pd.read_csv(self.root + 'raw/' + self.filename)
        if self.datasize is not None:
            if df.shape[0] > self.datasize:
                df = df.sample(
                    n=self.datasize,
                    random_state=self.random_state
                ).reset_index(drop=True)
        X = [self.mol2vec(smiles, fptype=self.fingerprint) for smiles in df['SMILES']]
        X = pd.DataFrame(X)
        X = X.dropna(axis=1)
        X = np.array(X.select_dtypes("number"))
        y = np.array(df.iloc[:,2])
        label = df['SMILES'].tolist()
        return X, y, label

                
def makedataset(
    data_mode,
    root,
    filename,
    r_max=8,
    n_neighbors=12,
    datasize=None,
    addH=False,
    random_state=0,
    fingerprint=None
):
    if data_mode == 'Crystal':
        dataset = CrystalDataset(root=root, 
                                 filename=filename,
                                 datasize=datasize,
                                 r_max=r_max,
                                 n_neighbors=n_neighbors,
                                 random_state=random_state)
        return dataset
    elif data_mode == 'Molecule':
        dataset = MoleculeDataset(root=root,
                                  filename=filename,
                                  datasize=datasize,
                                  addH=addH,
                                  random_state=random_state)
        return dataset
    elif data_mode == 'MoleculeFP':
        data = MoleculeFPDataset(
            root=root,
            filename=filename,
            datasize=datasize,
            fingerprint=fingerprint,
            random_state=random_state
        )
        X, y, label = data.file2data()
        return X, y, label


def datasplit(
    filepath,
    train_ratio,
    val_ratio,
    test_ratio,
    save_path_train,
    save_path_val,
    save_path_test,
    rand_split
):

    # Random split (fixed random state)
    df = pd.read_csv(filepath)
    df = df.sample(frac=1, ignore_index=True, random_state=rand_split)
    test_size = int(df.shape[0] * test_ratio)
    val_size = int((df.shape[0] - test_size) * val_ratio)
    df_test = df.iloc[:test_size, :].reset_index(drop=True)
    df_val = df.iloc[test_size:test_size+val_size, :].reset_index(drop=True)
    df_train = df.iloc[test_size+val_size:, :].reset_index(drop=True)

    # Save file
    df_train.to_csv(save_path_train, index=False)
    df_val.to_csv(save_path_val, index=False)
    df_test.to_csv(save_path_test, index=False)


def nn_data_process(X, y):
    X = torch.tensor(X, dtype = torch.float32)
    y = torch.tensor(y, dtype = torch.float32)
    dataset = TensorDataset(X, y)
    return dataset


# DELETE LATER
# def extract_data(dataset, model_name, fingerprint):
#     if model_name == 'RF':
#         X = [data.fingerprint[fingerprint] for data in dataset]
#         y = [data.y.item() for data in dataset]
#         label = [data.structure_id[0][0] for data in dataset]
#         return X, y, label

#     elif model_name == 'NN':
#         for data in dataset:
#             data.x = torch.tensor(data.fingerprint[fingerprint])
#             data.edge_index = None
#             data.edge_attr = None
#             data.edge_weight = None
#             data.u = None
#         return dataset

#     else:
#         for data in dataset:
#             data.fingerprint = None
#         return dataset
