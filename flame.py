import numpy as np
import pandas as pd

import os
import shutil
import time
import tqdm

from collections import Counter, defaultdict

import torch

from tensorboardX import SummaryWriter

class Trainer:
    def __init__(self, model, optimizer, criterion, device, scheduler=None, logdir=None):
        if device.type == 'cuda':
            self.model = model.cuda()
        else:
            self.model = model
            
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.logdir = None
        if logdir is not None:
            self.logdir = logdir
            if os.path.exists(self.logdir):
                shutil.rmtree(self.logdir)
            self.writer = SummaryWriter(log_dir=self.logdir)
            
        self.n_epoch = 1
        self.n_batch = 1
        self.n_starts = 0
        self.scheduler = scheduler
        
    def train_step(self, loader, cache=False, verbose=False):
        self.optimizer.zero_grad()
        self.model.train()
        
        if cache:
            preds = []
            targets = []
        total_loss = 0.
        with tqdm.tqdm(total=len(loader), disable=not verbose) as t:
            for obj, target in loader:
                obj, target = obj.to(self.device), target.to(self.device)
                pred = self.model(obj)
                if cache:
                    preds.append(pred.cpu().detach().numpy())
                    targets.append(target.cpu().detach().numpy())
                batch_loss = self.criterion(pred, target)
                batch_loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                t.set_description('Loss: {:.5f}. Iter'.format(batch_loss.item()))
                t.update()
                total_loss += batch_loss.item()
                if self.logdir is not None:
                    self.writer.add_scalar('loss_{}/batch_loss'.format(self.n_starts), batch_loss, self.n_batch)
                self.n_batch += 1
                
        total_loss /= len(loader.dataset)
        
        if cache:
            return total_loss, np.vstack(preds), np.vstack(targets)
        else:
            return total_loss
    
    def eval_step(self, loader, cache=False, verbose=False):
        self.model.eval()
        
        if cache:
            preds = []
        total_loss = 0.
        with tqdm.tqdm(total=len(loader), disable=not verbose) as t:
            for obj, target in loader:
                with torch.no_grad():
                    obj, target = obj.to(self.device), target.to(self.device)
                    pred = self.model(obj)
                    if cache:
                        preds.append(pred.cpu().detach().numpy())
                    batch_loss = self.criterion(pred, target)
                    total_loss += batch_loss.item()
                    t.set_description('Loss: {:.5f}. Iter'.format(batch_loss.item()))
                    t.update()
        total_loss /= len(loader.dataset)
        if cache:
            return total_loss, np.vstack(preds)
        else:
            return total_loss
    
    def train(self, train_loader, val_loader=None, n_epochs=3, checkpoints=False, verbose=False, actions=[]):
        self.n_starts += 1
        self.n_epoch = 1
        self.n_batch = 1
        if verbose:
            start = time.time()
        for n_epoch in range(1, n_epochs + 1):
            for action_epoch, action in actions:
                if action_epoch == n_epoch:
                    action(self.model)
            train_loss = self.train_step(train_loader)
            if verbose:
                print('{:3d} epoch. train: {:.5f}.'.format(n_epoch, train_loss), end=' ')
            if val_loader is not None:
                val_loss = self.eval_step(val_loader)
            if verbose:
                if val_loader is not None:
                    print('val: {:.5f}.'.format(val_loss), end=' ')
                print('time: {:.3f}s.'.format(time.time() - start))
            if self.logdir is not None:
                losses = {
                    'train': train_loss
                }
                if val_loader is not None:
                    losses['val'] = val_loss
                self.writer.add_scalars('loss_{}/epoch_loss'.format(self.n_starts), losses, self.n_epoch)
            self.n_epoch += 1
            if checkpoints and self.logdir is not None:
                torch.save(self.model.state_dict(), os.path.join(self.logdir, 'model_checkpoint_{}'.format(self.n_epoch)))

    def predict(self, dl):
        self.model.eval()
        preds = []
        for batch in dl:
            batch = batch.to(self.device)
            pred = self.model(batch)
            preds.append(pred.cpu().detach().numpy())
        preds = np.vstack(preds)
        return preds

