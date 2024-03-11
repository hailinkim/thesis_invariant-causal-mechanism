import sys
sys.path.append('/home/haikim20/angelicathesis')
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from src.utils.data_loader import combine_dataset, split_dataset, scale_dataset
from src.utils.config_generator import load_config
import torch.nn.functional as F
from src.model.model import Transformer
from src.model.train import train, test, validate
import math
import pickle
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import pandas as pd
import numpy as np
import random
import json
import datetime
import argparse
import os
from timeit import default_timer as timer

def causal_discovery(n_nodes, hyperparameter_file, model_path=None, test_mode=None):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load hyperparameters from JSON file
    config = load_config(hyperparameter_file)
    
    # Set the seed
    seed = config['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi-GPU.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
#     Combine data for n nodes
    ts, adj = combine_dataset(n_nodes) #subset can be specified
    
    # Train-Validate-Test Split
    ts_indices, ts_train, ts_validate, ts_test = split_dataset(ts, config['train_ratio'], config['val_ratio'], seed)
    adj_indices, adj_train, adj_validate, adj_test = split_dataset(adj, config['train_ratio'], config['val_ratio'], seed)
    
    assert torch.equal(ts_indices, adj_indices), "wrong split"
    
    # Min-max scale time series
    ts_train_normalized, ts_validate_normalized, ts_test_normalized = scale_dataset(ts_train, ts_validate, ts_test)
    
    # ts_train_normalized, adj_train = read_data()
    
    # Create TensorDataset objects
    train_dataset = TensorDataset(ts_train_normalized, adj_train)
    validate_dataset = TensorDataset(ts_validate_normalized, adj_validate)
    test_dataset = TensorDataset(ts_test_normalized, adj_test)

    # Load data
    trainloader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    valloader = DataLoader(validate_dataset, batch_size=1, shuffle=False)
    testloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    #check data shape
    ts, adj = next(iter(testloader))
    
    assert n_nodes +1 == ts.size(1) and n_nodes**2 == adj.size(1), "Incompatible data shape"
    src_seq_length = ts.size(2) #n_samples + 1
    
    model = Transformer(n_nodes, src_seq_length,
                        config['embed_dim'], 
                        config['n_layers_encoder'], 
                        config['n_layers_decoder'], 
                        config['n_heads'], 
                        config['n_heads_summary'], 
                        config['mlp_hidden_size'],
                        config['dropout'])   
    
    today = datetime.datetime.now().strftime("%Y%m%d")
    
    if model_path is not None:
        model.load_state_dict(torch.load(f'log/{model_path}'))
        print
    else:
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    model.to(device)

    criterion = nn.BCEWithLogitsLoss() #loss function
    optimizer = optim.Adam(model.parameters(), lr = config['learning_rate'], weight_decay = config["weight_decay"])
    scheduler = lr_scheduler.StepLR(optimizer, step_size=config['scheduler_step_size'], gamma=config['scheduler_gamma'])

    best_loss = float('inf')
    patience = config['patience_epoch']
    epochs_without_improvement = 0
    
    
    os.makedirs(f"/home/haikim20/angelicathesis/src/log/{today}", exist_ok=True)
    
    base_name_train = f'log/{today}/train_log_{today}'
    base_name_test = f'log/{today}/test_log_{today}'
    base_name_model = f'log/{today}/best_model_{today}'

    # Check if the file exists and create a unique name
    file_index = 0
    file_name_train = f"{base_name_train}.txt"
    file_name_test = f"{base_name_test}.txt"
    file_name_model = f'{base_name_model}.pth'
    
    while os.path.exists(file_name_train) or os.path.exists(file_name_test):
        file_index += 1
        file_name_train = f"{base_name_train}_{file_index}.txt"
        file_name_test = f"{base_name_test}_{file_index}.txt"
        file_name_model = f'{base_name_model}_{file_index}.pth'
    
    if test_mode is None:
        with open(file_name_train, 'w') as log:
            log.write('Model training log\n')
            log.write(f'Data: {n_nodes} nodes, {src_seq_length-1} samples\n')
            log.write(f'Hyperparameters used: {hyperparameter_file}\n')
            log.write('\n')

        for epoch in range(config["num_epochs"]):
            start_time = timer()
            train_loss = train(model, trainloader, optimizer, criterion)
            end_time = timer()
            validate_loss = validate(model, valloader, criterion)
            if (epoch + 1) % 10 == 0:
                print((f"Epoch: {epoch+1}, Train loss: {train_loss:.3f}, Val loss: {validate_loss:.3f}, "f"Epoch time = {(end_time - start_time):.3f}s"))

            if validate_loss < best_loss:
                best_loss = validate_loss
                best_epoch = epoch
                # Save the model
                torch.save(model.state_dict(), file_name_model)
                with open(file_name_train, 'a') as log:
                    log.write(f'Best epoch: {epoch}\n')
                    log.write(f'Best Validation Loss: {best_loss}\n')
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                print(f"Stopping early at epoch {epoch}, best loss: {best_loss}")
                break

            scheduler.step()
    
    print("testing started")
    tpr, fpr, fdr, hd = test(model, testloader)
    with open(file_name_test, 'a') as log:
        log.write(f'Testing on the model: {model_path} with hyperparameter: {hyperparameter_file}\n')
        log.write('Testing metrics: \n')
        log.write(f'TPR: {tpr}\n')
        log.write(f'FPR: {fpr}\n')
        log.write(f'FDR: {fdr}\n')
        log.write(f'HD: {hd}')
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a transformer model for temporal causal discovery.")
    parser.add_argument('--n_nodes', type=int, help='Number of variables')
    parser.add_argument('--config', type=str, help='Hyperparemter file name')
    parser.add_argument('--model_path', type=str, help='Model file path')
    parser.add_argument('--test_mode', type=str, help='Test only mode')

    args = parser.parse_args()
    n_nodes = args.n_nodes
    hyperparameter = args.config
    model_path = args.model_path
    test_mode = args.test_mode
    
    causal_discovery(n_nodes, hyperparameter, model_path, test_mode)
