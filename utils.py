import numpy as np

import time
from collections import defaultdict

import torch
from torch.utils.data import TensorDataset, DataLoader, Dataset

from metrics import lwlrap_score
from flame import Trainer

def create_ds(X, y=None):
    if y is not None:
        
        ds = TensorDataset(
            torch.tensor(X) \
                .permute(0, 2, 1) \
                .unsqueeze(1) \
                .contiguous(),
            torch.tensor(y) \
                .float()
        )
    else:
        
        ds = TensorDataset(
            torch.tensor(X) \
                .permute(0, 2, 1) \
                .unsqueeze(1) \
                .contiguous(),
        )
        
    return ds


def train_model(trainer, train_dl, val_dl, val_y, scheduler, n_epochs, gap=None, verbose=False):
    epoch_times = []
    
    losses = {
        'train': defaultdict(list),
        'val': defaultdict(list)
    }

    print('Epoch, best_val_lwlrap')
    best_loss = -1
    patience = 0
    i = 0
    while patience < gap and i <= n_epochs:
        start = time.time()
        train_logloss, train_preds, train_y = trainer.train_step(train_dl, cache=True)
        epoch_times.append(time.time() - start)
        train_score = lwlrap_score(train_y, train_preds)
        
        val_logloss, val_preds = trainer.eval_step(val_dl, cache=True)
        val_score = lwlrap_score(val_y, val_preds)
        
        if scheduler is not None:
            scheduler.step(val_score, epoch=i + 1)
        
        if best_loss < val_score:
            patience = 0
            best_loss = val_score
            if verbose:
                print('{:>4}, {:.4f}'.format(i + 1, best_loss))
        else:
            patience += 1
            
        losses['train']['logloss'].append(train_logloss)
        losses['train']['lwlrap'].append(train_score)
        losses['val']['logloss'].append(val_logloss)
        losses['val']['lwlrap'].append(val_score)
        
        i += 1
    
    if verbose:
        print('Done.')

    return losses, best_loss, np.mean(epoch_times)
