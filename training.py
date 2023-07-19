import itertools
import optuna
import gc
import torch
import numpy as np
from torch import optim
from model import makemodel
from sklearn.utils import shuffle
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from torch_geometric.loader import DataLoader
from sklearn.manifold import TSNE


# GNN training
##########################################
def eval(loader, loss_fn, model, cuda_id):
    y_exp = torch.tensor([])
    y_pred = torch.tensor([])
    label = []
    for data in loader:
        data = data.cuda(cuda_id)
        out = model(data)
        y_exp = torch.cat((y_exp, torch.reshape(data.y.cpu(), (-1,))))
        y_pred = torch.cat((y_pred, out.cpu()))
        label += list(itertools.chain.from_iterable(
            list(itertools.chain.from_iterable(data.structure_id))
        ))
        del data, out
        torch.cuda.empty_cache()
    loss = loss_fn(y_exp, y_pred)

    return loss, y_exp, y_pred, label


def train(loader_train, loader_test, model, n_epoch, lr, cuda_id):
    # loss function
    loss_fn = torch.nn.L1Loss()
    record_loss_test = []
    record_loss_train = []
    model.cuda(cuda_id)

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(n_epoch):
        # train mode
        model.train()
        for data in loader_train:
            data = data.cuda(cuda_id)
            optimizer.zero_grad()        
            out = model(data)              
            loss = loss_fn(out, torch.reshape(data.y, (-1,))) 
            loss.backward()              
            optimizer.step()
            del data, out, loss
            torch.cuda.empty_cache()
            # gc.collect()

        # predict mode
        model.eval()
        with torch.no_grad():
            (loss_train,
             y_train_exp,
             y_train_pred,
             label_train) = eval(loader_train, loss_fn, model, cuda_id)

            (loss_test,
             y_test_exp,
             y_test_pred,
             label_test) = eval(loader_test, loss_fn, model, cuda_id)
        
        record_loss_train.append(loss_train.item())
        record_loss_test.append(loss_test.item())
        # torch.cuda.empty_cache()

        if (epoch+1)%10==0:
            print("Epoch:", epoch+1,
                  "MAE_train: %.3f" % loss_train.item(),
                  "MAE_test: %.3f" % loss_test.item(),)
        # if (epoch+1)%50==0:
        #     memory_allocated = torch.cuda.memory_allocated(device=cuda_id)/10**9
        #     memory_reserved = torch.cuda.memory_reserved(device=cuda_id)/10**9
        #     print(f'Epoch: {epoch+1},'\
        #           f'allocated {memory_allocated:.3f} GB,'\
        #           f'reserved {memory_reserved:.3f} GB')
            
    return (record_loss_train,
            record_loss_test,
            y_train_exp.tolist(),
            y_train_pred.tolist(),
            y_test_exp.tolist(),
            y_test_pred.tolist(),
            label_train,
            label_test,
            model)


def predict(loader_test, model, cuda_id):
    loss_fn = torch.nn.L1Loss()
    model.cuda(cuda_id)
    model.eval()
    with torch.no_grad():
        (loss_test,
         y_test_exp,
         y_test_pred,
         label_test) = eval(loader_test, loss_fn, model, cuda_id)
        
    return y_test_exp.tolist(), y_test_pred.tolist(), label_test


# NN training
####################################################
def eval_nn(loader, loss_fn, model, cuda_id):
    y_exp = torch.tensor([])
    y_pred = torch.tensor([])
    label = []
    for X, y in loader:
        X = X.cuda(cuda_id)
        y = y.cuda(cuda_id)
        out = model(X)
        y_exp = torch.cat((y_exp, torch.reshape(y.cpu(), (-1,))))
        y_pred = torch.cat((y_pred, out.cpu()))
        del X, y, out
        torch.cuda.empty_cache()
    loss = loss_fn(y_exp, y_pred)

    return loss, y_exp, y_pred


def train_nn(loader_train, loader_test, model, n_epoch, lr, cuda_id):
    # loss function
    loss_fn = torch.nn.L1Loss()
    record_loss_test = []
    record_loss_train = []
    model.cuda(cuda_id)

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(n_epoch):
        # train mode
        model.train()
        for X, y in loader_train:
            X = X.cuda(cuda_id)
            y = y.cuda(cuda_id)
            optimizer.zero_grad()        
            out = model(X)              
            loss = loss_fn(out, torch.reshape(y, (-1,))) 
            loss.backward()              
            optimizer.step()
            del X, y, out, loss
            torch.cuda.empty_cache()
            gc.collect()

        # predict mode
        model.eval()
        (loss_train,
         y_train_exp,
         y_train_pred) = eval_nn(loader_train, loss_fn, model, cuda_id)
        (loss_test,
         y_test_exp,
         y_test_pred) = eval_nn(loader_test, loss_fn, model, cuda_id)

        record_loss_train.append(loss_train.item())
        record_loss_test.append(loss_test.item())
        torch.cuda.empty_cache()        

        if (epoch+1)%10==0:
            print("Epoch:", epoch+1,
                  "MAE_train: %.3f" % loss_train.item(),
                  "MAE_test: %.3f" % loss_test.item(),)
            
    return (record_loss_train,
            record_loss_test,
            y_train_exp.tolist(),
            y_train_pred.tolist(),
            y_test_exp.tolist(),
            y_test_pred.tolist(),
            model)


def predict_nn(loader_test, model, cuda_id):
    loss_fn = torch.nn.L1Loss()
    model.cuda(cuda_id)
    model.eval()
    (loss_test,
     y_test_exp,
     y_test_pred) = eval_nn(loader_test, loss_fn, model, cuda_id)
        
    return y_test_exp.tolist(), y_test_pred.tolist()


###############################################
def train_rf(X_train, X_test, y_train, y_test, model):
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    return y_train_pred, y_test_pred, model


###################################################################
class Objective:
    def __init__(self, dataset, loader_train, loader_val, model, n_epoch, cuda_id):
        self.dataset = dataset
        self.loader_train = loader_train
        self.loader_val = loader_val
        self.model = model
        self.n_epoch = n_epoch
        self.cuda_id = cuda_id

    def __call__(self, trial):
        # Search space
        model_params = {
            'data': self.dataset,
            "dim1": trial.suggest_int('dim1', 10, 250, 20),
            "dim2": trial.suggest_int('dim2', 10, 250, 20),
            "dim3": trial.suggest_int('dim3', 10, 250, 20),
            "gnn_count": trial.suggest_int('gnn_count', 1, 9),
            "post_fc_count": trial.suggest_int('post_fc_count', 1, 9),
            "pool": trial.suggest_categorical(
                'pool',
                ["global_mean_pool", "global_add_pool", "global_max_pool", "set2set"]
            ),
        }
        model = makemodel(self.model, **model_params)

        # Loss function
        loss_fn = torch.nn.L1Loss()
        model.cuda(self.cuda_id)

        # Optimizer
        lr = trial.suggest_categorical('lr', [1e-4, 5e-4, 1e-3, 5e-3, 1e-2])
        optimizer = optim.AdamW(model.parameters(), lr=lr)

        for epoch in range(self.n_epoch):
            model.train()
            for data in self.loader_train: 
                data = data.cuda(self.cuda_id)
                optimizer.zero_grad()        
                out = model(data)  
                loss = loss_fn(out, torch.reshape(data.y, (-1,)))
                loss.backward()
                optimizer.step()
                torch.cuda.empty_cache()
            if (epoch+1)%20==0:
                print(f'epoch {epoch+1} done')

            model.eval()
            with torch.no_grad():
                (loss_val,
                 y_val_exp,
                 y_val_pred,
                 label_val) = eval(self.loader_val, loss_fn, model, self.cuda_id)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

        return loss_val


def hyperparameter(init_data, loader_train, loader_val, model, n_epoch, save_name, cuda_id):
    objective = Objective(init_data, loader_train, loader_val, model, n_epoch, cuda_id)
    sampler = optuna.samplers.TPESampler(seed=0)
    study = optuna.create_study(sampler=sampler, 
                                direction='minimize',
                                storage=f"sqlite:///{save_name}.db",
                                study_name=save_name,
                                load_if_exists=True)
    study.optimize(objective, n_trials=50, gc_after_trial=True)
    return study


###############################################################
# Hyperparameter optim for fingerprint
class ObjectiveNN:
    def __init__(self, data, loader_train, loader_val, model, n_epoch, cuda_id):
        self.data = data
        self.loader_train = loader_train
        self.loader_val = loader_val
        self.model = model
        self.n_epoch = n_epoch
        self.cuda_id = cuda_id

    def __call__(self, trial):
        # Search space
        model_params = {
            'data': self.data,
            'n_layers': trial.suggest_int('n_layers', 1, 9),
            'n_dim': trial.suggest_int('n_dim', 50, 300, 50),
        }
        model = makemodel(self.model, **model_params)

        # Loss function
        loss_fn = torch.nn.L1Loss()
        model.cuda(self.cuda_id)

        # Optimizer
        lr = trial.suggest_categorical('lr', [1e-4, 5e-4, 1e-3, 5e-3, 1e-2])
        optimizer = optim.AdamW(model.parameters(), lr=lr)

        for epoch in range(self.n_epoch):
            model.train()
            for X, y in self.loader_train: 
                X = X.cuda(self.cuda_id)
                y = y.cuda(self.cuda_id)
                optimizer.zero_grad()        
                out = model(X)  
                loss = loss_fn(out, torch.reshape(y, (-1,)))
                loss.backward() 
                optimizer.step()
                torch.cuda.empty_cache()

            model.eval()
            (loss_val,
             y_val_exp,
             y_val_pred) = eval_nn(self.loader_val, loss_fn, model, self.cuda_id)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
        
        return loss_val


def hyperparameter_nn(init_data, loader_train, loader_val, model, n_epoch, save_name, cuda_id):
    objective = ObjectiveNN(init_data, loader_train, loader_val, model, n_epoch, cuda_id)
    sampler = optuna.samplers.TPESampler(seed=0)
    study = optuna.create_study(sampler=sampler, 
                                direction='minimize',
                                storage=f"sqlite:///{save_name}.db",
                                study_name=save_name,
                                load_if_exists=True)
    study.optimize(objective, n_trials=5, gc_after_trial=True)
    return study


#################################################
class ObjectiveRF:
    def __init__(self, X_train, X_val, y_train, y_val):
        self.X_train = X_train
        self.X_val = X_val
        self.y_train = y_train
        self.y_val = y_val

    def __call__(self, trial):
        model_params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300, 50),
            'max_features': trial.suggest_int('max_features', 10, 200, 10),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20, 2),
            'max_depth': trial.suggest_int('max_depth', 20, 200, 20),
        }
        model = RandomForestRegressor(**model_params)
        model.fit(self.X_train, self.y_train)
        y_pred_val = model.predict(self.X_val)
        loss_val = mean_absolute_error(self.y_val, y_pred_val)
        return loss_val


def hyperparameter_rf(X_train, X_val, y_train, y_val, save_name):
    objective = ObjectiveRF(X_train, X_val, y_train, y_val)
    sampler = optuna.samplers.TPESampler(seed=0)
    study = optuna.create_study(sampler=sampler, 
                                direction='minimize',
                                storage=f"sqlite:///{save_name}.db",
                                study_name=save_name,
                                load_if_exists=True)
    study.optimize(objective, n_trials=10, gc_after_trial=True)
    return study
    

########################################
def gradcam(loader, model, cuda_id):
    model.train()
    heat_trial = 0
    model.gradcam = True
    for data in loader:
        data = data.cuda(cuda_id)
        predict = model(data)
        next(iter(predict)).backward()
        gradients = model.get_activations_gradient()
        pooled_gradients = torch.mean(gradients, dim=[0])
        activations = model.get_activations(data).detach()
        for i in range(activations.shape[-1]):
            activations[:, i] *= pooled_gradients[i]

        if heat_trial == 0:
            heatmap = torch.mean(activations, dim=1).squeeze().cpu().numpy()
        else:
            heatmap_temp = torch.mean(activations, dim=1).squeeze().cpu().numpy()
            heatmap = np.concatenate([heatmap, heatmap_temp], axis=0)
        heat_trial += 1
    print("heatmap: ", heatmap.shape)

    return heatmap


def tsne(loader, model, cuda_id):
    model.eval()
    model.manifold = True
    mani = TSNE()
    X = []
    label = []
    for data in loader:
        data = data.cuda(cuda_id)
        out = model(data)
        X += model.feat_manifold.cpu().detach().tolist()
        label += list(itertools.chain.from_iterable(
            list(itertools.chain.from_iterable(data.structure_id))
        ))
        del data, out
        torch.cuda.empty_cache()
    print('model.feat_manifold.size()', model.feat_manifold.size())
    print('len(X)', len(X))
    X = np.array(X)
    print('X.shape', X.shape)
    X = mani.fit_transform(X)
    print('X.shape', X.shape)
    return X, label