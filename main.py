import os
from process import *
from training import *
from model import *
import torch
import settings
import time
import pickle


def main(
    ##########
    work_name,
    run_mode,
    data_mode,
    model_name,
    train_size,
    fingerprint,
    random_state,
    model_path,
    load_model,
    cuda_id,
    ###########
    root,
    root_train,
    root_val,
    root_test,
    file_name_raw,
    file_name_train,
    file_name_val,
    file_name_test,
    ###########
    graph_max_radius,
    graph_max_neighbors,
    addH,
    ##########
    train_ratio,
    val_ratio,
    test_ratio,
    rand_split,
    split_data,
    ##########
    batch_size,
    n_epoch,
    lr,
    #########
    dim1,
    dim2,
    dim3,
    pre_fc_count,
    gc_count,
    gc_fc_count,
    post_fc_count,
    pool,
    pool_order,
    batch_norm,
    batch_track_stats,
    act,
    dropout_rate,
    ########
    n_dim,
    n_layers,
    ########
    n_estimators,
    max_features,
    min_samples_leaf,
    max_depth,
):
    start_time = time.time()
    print("Starting...")
    print(
        "GPU is available:",
        torch.cuda.is_available(),
        ", Quantity: ",
        torch.cuda.device_count(),
    )
    
    # Set name
    root_train = root + root_train
    root_val = root + root_val
    root_test = root + root_test
    save_path = root + 'results/'
    hyper_path = root + 'hyperopt/'
    train_size = int(train_size)
    val_size = None # int(val_ratio*data_size)
    test_size = None # int(test_ratio*data_size)
    columns_loss = ['train loss', 'test loss']
    columns_pred = ['ID', 'exp', 'pred']

    if data_mode == 'Crystal':
        name_crystal = f'{data_mode}_r{graph_max_radius}_n{graph_max_neighbors}'
        folder_train = f'processed_{name_crystal}_N{train_size}_rs{random_state}'
        folder_val = f'processed_{name_crystal}'
        folder_test = f'processed_{name_crystal}'

    elif data_mode == 'Molecule':
        name_molecule = f'{data_mode}_H{addH}'
        folder_train = f'processed_{name_molecule}_N{train_size}_rs{random_state}'
        folder_val = f'processed_{name_molecule}'
        folder_test = f'processed_{name_molecule}'
    
    # Preparation (first data split)
    if split_data is True:
        if (os.path.exists(root_train) is False and
        os.path.exists(root_val) is False and
        os.path.exists(root_test) is False):
            os.makedirs(root_train + 'raw/')
            os.makedirs(root_val + 'raw/')
            os.makedirs(root_test + 'raw/')
            datasplit(root + file_name_raw,
                      train_ratio,
                      val_ratio,
                      test_ratio,
                      root_train + 'raw/' + file_name_train,
                      root_val + 'raw/' + file_name_val,
                      root_test + 'raw/' + file_name_test,
                      rand_split)

    # Check data mode
    if data_mode != 'Crystal' and data_mode != 'Molecule' and data_mode != 'MoleculeFP':
        print('Please choose valid data mode')
        
    # Crystal & Molecular Graph mode
    if data_mode == 'Crystal' or data_mode == 'Molecule':
        if run_mode=='Training':

            # Prepare
            print('Training mode start')
            if (os.path.exists(root_train + 'processed/') is False and
            os.path.exists(root_val + 'processed/') is False and
            os.path.exists(root_test + 'processed/') is False):
                os.makedirs(root_train + folder_train, exist_ok=True)
                os.makedirs(root_test + folder_test, exist_ok=True)
            try:
                os.rename(root_train + folder_train, root_train + 'processed/')
                os.rename(root_test + folder_test, root_test + 'processed/')
            except:
                pass

            # Make graphs
            dataset_train = makedataset(data_mode,
                                        root_train,
                                        file_name_train,
                                        graph_max_radius,
                                        graph_max_neighbors,
                                        train_size,
                                        addH,
                                        random_state)
            dataset_test = makedataset(data_mode,
                                       root_test,
                                       file_name_test,
                                       graph_max_radius,
                                       graph_max_neighbors,
                                       test_size,
                                       addH,
                                       random_state)

            loader_train = DataLoader(dataset_train, batch_size, shuffle=True)
            loader_test = DataLoader(dataset_test, batch_size, shuffle=False)

            model_params = {
                'data': dataset_test,
                'dim1': dim1,
                'dim2': dim2,
                'dim3': dim3,
                'gnn_count': gc_count,
                'post_fc_count': post_fc_count,
                'pool': pool
            }

            model = makemodel(model_name, **model_params)
            if load_model != None:
                # model.load_state_dict(torch.load(root_train+load_model))
                model = torch.load(model_path + load_model)
                print(f'Existing model {load_model} loaded')
            (record_loss_train,
            record_loss_test,
            y_train_exp,
            y_train_pred,
            y_test_exp,
            y_test_pred,
            label_train,
            label_test,
            model) = train(loader_train, loader_test, model, n_epoch, lr, cuda_id)

            ## Save results
            os.makedirs(save_path, exist_ok=True)

            results_train = [label_train, y_train_exp, y_train_pred]
            results_test = [label_test, y_test_exp, y_test_pred]
            train_pred = pd.DataFrame(results_train, index=columns_pred).T
            test_pred = pd.DataFrame(results_test, index=columns_pred).T
            
            if data_mode == 'Crystal':
                save_name = f'{work_name}_Ntrain{train_size}_rs{random_state}_{model_name}'\
                f'_gc{gc_count}_post{post_fc_count}'
            elif data_mode == 'Molecule':
                save_name = f'{work_name}_Ntrain{train_size}_rs{random_state}_{model_name}'\
                f'_gc{gc_count}_post{post_fc_count}'

            results_loss = [record_loss_train, record_loss_test]
            loss = pd.DataFrame(results_loss, index=columns_loss).T
            loss.to_csv(save_path + save_name + '_loss.csv', index=False)
            torch.save(model, save_path + save_name + '.pth')
            
            train_pred.to_csv(save_path + save_name + '_train.csv', index=False)
            test_pred.to_csv(save_path + save_name + '_test.csv', index=False)
            try:
                os.rename(root_train + 'processed', root_train + folder_train)
                os.rename(root_test + 'processed', root_test + folder_test)
            except:
                pass

        elif run_mode == 'Predict':
            print('Predict mode start')
            if os.path.exists(root_test + 'processed/') is False:
                os.makedirs(root_test + folder_test, exist_ok=True)
            try:
                os.rename(root_test + folder_test, root_test + 'processed/')
            except:
                pass
            dataset_test = makedataset(
                data_mode=data_mode,
                root=root_test,
                filename=file_name_test,
                r_max=graph_max_radius,
                n_neighbors=graph_max_neighbors,
                addH=addH,
            )
            loader_test = DataLoader(dataset_test, batch_size, shuffle=False)
            model = torch.load(model_path + load_model)
            y_test_exp, y_test_pred, label_test = predict(loader_test, model, cuda_id)
            test_pred = pd.DataFrame([label_test, y_test_exp, y_test_pred], index=columns_pred).T
            os.makedirs(save_path, exist_ok=True)
            test_pred.to_csv(save_path + work_name + load_model + '_predict.csv', index=False)
            try:
                os.rename(root_test + 'processed', root_test + folder_test)
            except:
                pass

        elif run_mode == 'Explain':
            print('Explain mode start')
            os.makedirs(root_test + folder_test, exist_ok=True)
            try:
                os.rename(root_test + folder_test, root_test + 'processed')
            except:
                pass

            # Make graphs
            dataset_test = makedataset(
                data_mode=data_mode,
                root=root_test,
                filename=file_name_test,
                r_max=graph_max_radius,
                n_neighbors=graph_max_neighbors,
                addH=addH,
            )
            loader_test = DataLoader(dataset_test, batch_size, shuffle=False)
            model = torch.load(model_path + load_model)
            heatmap = gradcam(loader_test, model, cuda_id)
            manifold, label_test = tsne(loader_test, model, cuda_id)
            np.savetxt(save_path + load_model + '_explain_gradcam.csv', heatmap)
            result_mani = pd.DataFrame([label_test, manifold[:,0], manifold[:,1]], index=['ID', 'dim1', 'dim2']).T
            result_mani.to_csv(save_path + load_model + '_explain_tsne.csv', index=False)
            try:
                os.rename(root_test + 'processed', root_test + folder_test)
            except:
                pass

        elif run_mode == 'Hyperopt':
            print('Hyperopt mode start')
            os.makedirs(root_train + folder_train, exist_ok=True)
            os.makedirs(root_val + folder_val, exist_ok=True)
            try:
                os.rename(root_train + folder_train, root_train + 'processed')
                os.rename(root_val + folder_val, root_val + 'processed')
            except:
                pass

            # Make graphs
            dataset_train = makedataset(data_mode,
                                        root_train,
                                        file_name_train,
                                        graph_max_radius,
                                        graph_max_neighbors,
                                        train_size,
                                        addH,
                                        random_state)
            dataset_val = makedataset(data_mode,
                                      root_val,
                                      file_name_val,
                                      graph_max_radius,
                                      graph_max_neighbors,
                                      val_size,
                                      addH,
                                      random_state)
            loader_train = DataLoader(dataset_train, batch_size, shuffle=True)
            loader_val = DataLoader(dataset_val, batch_size, shuffle=False)

            # Hyperparameter search
            os.makedirs(hyper_path, exist_ok=True)
            save_name = f'{hyper_path}{work_name}_{run_mode}_Ntrain{train_size}'
            study = hyperparameter(dataset_val,
                                   loader_train,
                                   loader_val,
                                   model_name,
                                   n_epoch,
                                   save_name,
                                   cuda_id)
            print('params:', study.best_params)
            hist_df = study.trials_dataframe(multi_index=True)
            hist_df.to_csv(f'{save_name}.csv', index=False)
            try:
                os.rename(root_train + 'processed', root_train + folder_train)
                os.rename(root_val + 'processed', root_val + folder_val)
            except:
                pass

        else:
            print('Please choose valid run_mode')
    
    ###############################################
    elif data_mode == 'MoleculeFP':
        if run_mode == 'Training':
            print('Training mode start')
            X_train, y_train_exp, label_train = makedataset(
                data_mode=data_mode,
                root=root_train,
                filename=file_name_train,
                datasize=train_size,
                random_state=random_state,
                fingerprint=fingerprint,
            )
            X_test, y_test_exp, label_test = makedataset(
                data_mode=data_mode,
                root=root_test,
                filename=file_name_test,
                datasize=test_size,
                random_state=random_state,
                fingerprint=fingerprint
            )
            if model_name == 'RF':
                model_params = {
                        'n_estimators': n_estimators,
                        'max_features': max_features,
                        'min_samples_leaf': min_samples_leaf,
                        'max_depth': max_depth
                    }
                model = RF(**model_params)
                (y_train_pred,
                 y_test_pred, 
                 model) = train_rf(X_train, X_test, y_train_exp, y_test_exp, model)
                os.makedirs(save_path, exist_ok=True)
                save_name = f'{work_name}_{data_mode}_{fingerprint}_Ntrain{train_size}_rs{random_state}_{model_name}'
                pickle.dump(model, open(save_path + save_name + '.pkl', 'wb'))

            elif model_name == 'NN':
                dataset_train = nn_data_process(X_train, y_train_exp)
                dataset_test  = nn_data_process(X_test, y_test_exp)
                loader_train = DataLoader(dataset_train, batch_size, shuffle=True)
                loader_test = DataLoader(dataset_test, batch_size, shuffle=False)
                model_params = {
                    'data': X_train,
                    'n_dim': n_dim,
                    'n_layers': n_layers,
                }
                model = makemodel(model_name, **model_params)
                (record_loss_train,
                record_loss_test,
                y_train_exp,
                y_train_pred,
                y_test_exp,
                y_test_pred,
                model) = train_nn(loader_train, loader_test, model, n_epoch, lr, cuda_id)
                
                save_name = f'{work_name}_{data_mode}_{fingerprint}_Ntrain{train_size}_rs{random_state}'\
                f'{model_name}_fc{n_layers}_dim{n_dim}'
                
                ## Save results
                os.makedirs(save_path, exist_ok=True)
                results_loss = [record_loss_train, record_loss_test]
                loss = pd.DataFrame(results_loss, index=columns_loss).T
                loss.to_csv(save_path + save_name + '_loss.csv', index=False)
                torch.save(model, save_path + save_name + '.pth')

            results_train = [label_train, y_train_exp, y_train_pred]
            results_test = [label_test, y_test_exp, y_test_pred]
            train_pred = pd.DataFrame(results_train, index=columns_pred).T
            test_pred = pd.DataFrame(results_test, index=columns_pred).T
            train_pred.to_csv(save_path + save_name + '_train.csv', index=False)
            test_pred.to_csv(save_path + save_name + '_test.csv', index=False)
        
        elif run_mode == 'Predict':
            print('Predict mode start')
            X_test, y_test_exp, label_test = makedataset(
                data_mode=data_mode,
                root=root_test,
                filename=file_name_test,
                fingerprint=fingerprint
            )
            if model_name == 'NN':
                dataset_test  = nn_data_process(X_test, y_test_exp)
                loader_test = DataLoader(dataset_test, batch_size, shuffle=False)
                model = torch.load(model_path + load_model)
                y_test_exp, y_test_pred = predict_nn(loader_test, model, cuda_id)
                
            elif model_name == 'RF':
                model = pickle.load(open(model_path + load_model, 'rb'))
                y_test_pred = model.predict(X_test)
            
            test_pred = pd.DataFrame([label_test, y_test_exp, y_test_pred], index=columns_pred).T
            test_pred.to_csv(save_path + load_model + '_predict.csv', index=False)
            

        elif run_mode == 'Hyperopt':
            print('Hyperopt mode start')
            os.makedirs(hyper_path, exist_ok=True)
            save_name = f'{hyper_path}{work_name}_{run_mode}_{data_mode}_{fingerprint}_{model_name}'

            X_train, y_train_exp, label_train = makedataset(
                data_mode=data_mode,
                root=root_train,
                filename=file_name_train,
                datasize=train_size,
                random_state=random_state,
                fingerprint=fingerprint,
            )
            X_val, y_val_exp, label_val = makedataset(
                data_mode=data_mode,
                root=root_val,
                filename=file_name_val,
                datasize=val_size, # CHANGE LATER int(val_ratio*data_size)
                random_state=random_state,
                fingerprint=fingerprint
            )
            if model_name == 'NN':
                dataset_train = nn_data_process(X_train, y_train_exp)
                dataset_val  = nn_data_process(X_val, y_val_exp)
                loader_train = DataLoader(dataset_train, batch_size, shuffle=True)
                loader_val = DataLoader(dataset_val, batch_size, shuffle=False)
                study = hyperparameter_nn(
                    X_train,
                    loader_train,
                    loader_val,
                    model_name,
                    n_epoch,
                    save_name,
                    cuda_id
                )                
            elif model_name == 'RF':                
                study = hyperparameter_rf(X_train, X_val, y_train_exp, y_val_exp, save_name)

            print('params:', study.best_params)
            hist_df = study.trials_dataframe(multi_index=True)
            hist_df.to_csv(f'{save_name}.csv', index=False)
            
    # Finishing
    total = time.time() - start_time
    minute = total//60
    second = total%60
    print(f'--- {int(minute)} min {int(second)} sec elapsed ---')


if __name__ == "__main__":
    params = {
        ###########
        'work_name': settings.work_name,
        'run_mode': settings.run_mode,
        'data_mode': settings.data_mode,
        'model_name': settings.model_name,
        'train_size': settings.train_size,
        'fingerprint': settings.fingerprint,
        'random_state': settings.random_state,
        'model_path': settings.model_path,
        'load_model': settings.load_model,
        'cuda_id': settings.cuda_id,
        ###########
        'root': settings.root,
        'root_train': settings.root_train,
        'root_val': settings.root_val,
        'root_test': settings.root_test,
        'file_name_raw': settings.file_name_raw,
        'file_name_train': settings.file_name_train,
        'file_name_val': settings.file_name_val,
        'file_name_test': settings.file_name_test,
        ##########
        'graph_max_radius': settings.graph_max_radius,
        'graph_max_neighbors': settings.graph_max_neighbors,
        'addH': settings.addH,
        ###########
        'train_ratio': settings.train_ratio,
        'val_ratio': settings.val_ratio,
        'test_ratio': settings.test_ratio,
        'rand_split': settings.rand_split,
        'split_data': settings.split_data,
        ##########
        'batch_size': settings.batch_size,
        'n_epoch': settings.n_epoch,
        'lr': settings.lr,
        ##########
        'dim1': settings.dim1,
        'dim2': settings.dim2,
        'dim3': settings.dim3,
        'pre_fc_count': settings.pre_fc_count,
        'gc_count': settings.gc_count,
        'gc_fc_count': settings.gc_fc_count,
        'post_fc_count': settings.post_fc_count,
        'pool': settings.pool,
        'pool_order': settings.pool_order,
        'batch_norm': settings.batch_norm,
        'batch_track_stats': settings.batch_track_stats,
        'act': settings.act,
        'dropout_rate': settings.dropout_rate,
        ##########
        'n_dim': settings.n_dim,
        'n_layers': settings.n_layers,
        ##########
        'n_estimators': settings.n_estimators,
        'max_features': settings.max_features,
        'min_samples_leaf': settings.min_samples_leaf,
        'max_depth': settings.max_depth,
    }
    main(**params)
