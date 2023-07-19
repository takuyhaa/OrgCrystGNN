# Specipy work details
work_name = 'Large-screening_TUWGEX-' # Work name for save
run_mode = 'Training' # {Training, Predict, Explain, Hyperopt}
data_mode = 'Crystal' # {Crystal, Molecule}
model_name = 'MEGNet' # {MEGNet, GCN, CGCNN, SchNet}
train_size = 6000 #8457
fingerprint = 'Avalon'
random_state = 2
model_path = None
load_model = None
cuda_id = 'cuda:0'

# Specify working directory
root = 'D:datasets/Bandgap_rev/'
root_train = 'trainset/'
root_val = 'valset/'
root_test = 'testset/'
file_name_raw = 'smiles-cod-bandgap.csv'
file_name_train = 'dataset_train.csv'
file_name_val = 'dataset_val.csv'
file_name_test = 'dataset_test.csv'

# Specify graph parameters
graph_max_radius = 8
graph_max_neighbors = 12
addH = False

# Initial data split
train_ratio = 0.8
val_ratio = 0.05
test_ratio = 0.15
rand_split = 1
split_data = False

# Training
batch_size = 64 #No change
n_epoch = 200 #No change
lr = 0.001

# GNN Model
dim1 = 90
dim2 = 70
dim3 = 210
pre_fc_count = 1 #No change
gc_count = 3
gc_fc_count = 1 #No change
post_fc_count = 2
pool = "global_max_pool"
pool_order = "early" #No change
batch_norm = "True" #No change
batch_track_stats = "True" #No change
act = "relu" #No change
dropout_rate = 0.0 #No change

# NN model
n_dim = 100
n_layers = 3

# RF model
n_estimators = 200
max_features = 100
min_samples_leaf = 5
max_depth = 30