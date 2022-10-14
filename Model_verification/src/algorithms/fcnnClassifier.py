import os
import time
import datetime
import copy
import gc
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.model_selection import ShuffleSplit
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd

from .neural_net import neural_net
from .neural_net import smallest_k_dist_loss
from .neural_net import IntermediateSequential


class fcnnClassifier(neural_net):
    def __init__(
        self,
        topology: list,
        name: str = "fcnnClassifier",
        dropout: float = 0,
        L2Reg: float = 0,
        num_epochs: int = 100,
        batch_size: int = 20,
        lr: float = 1e-3,
        seed: int = None,
        save_interm_models = False,
    ):
        super().__init__(name=name, num_epochs=num_epochs,
                batch_size=batch_size, lr=lr, seed=seed,
                save_interm_models=save_interm_models)
        self.topology = topology
        self.dropout = dropout
        self.L2Reg = L2Reg
        self.module = fcnnClassifierModule(self.topology, self.dropout)

    def fit(self, dataset, run_folder=None, logger=None):
        train_data = dataset.train_data()
        labels = dataset.train_labels
        joined_trained_data = pd.concat([train_data, labels], axis=1)
        data_loader = DataLoader(
            num_workers=0,
            dataset=joined_trained_data.values,
            batch_size=self.batch_size,
            shuffle = True,
            drop_last=True,
            pin_memory=True,
        )
        #data_loader = self.get_data_loader(X)
        optimizer = torch.optim.Adam(
            params=self.module.parameters(), lr=self.lr, weight_decay=self.L2Reg
        )
        loss_type = nn.CrossEntropyLoss()
        self.module.train()
        for epoch in range(self.num_epochs):
            epoch_start = datetime.datetime.now()
            epoch_loss = 0
            i = 0
            data_len = len(data_loader)
            for inst_batch in data_loader:
                #train_batch = inst_batch
                train_batch = inst_batch[:,:-1]
                label_batch = inst_batch[:,-1]
                output = self.module(train_batch)[0]
                i = i + 1
                loss = loss_type(output, label_batch.long())
                self.module.zero_grad()
                epoch_loss += loss
                loss.backward()
                optimizer.step()
            epoch_end = datetime.datetime.now()
            duration = epoch_end - epoch_start
            if run_folder and self.save_interm_models:
                self.save(run_folder, subfolder=f'Epochs/Epoch_{epoch}')
            else:
                self.save(run_folder)
            if logger:
                logger.info(f"Training epoch {epoch}")
                logger.info(f"Epoch_loss: {epoch_loss}")
                logger.info(f"Duration: {duration}")
        self.module.erase_dropout()

    def predict(self, X: pd.DataFrame):
        self.module.eval()
        data_loader = DataLoader(
            dataset=X.values,
            batch_size=self.batch_size,
            drop_last=False,
            pin_memory=True,
        )
        outputs = []
        for inst_batch in data_loader:
            inst_batch = inst_batch.float()
            outputs.append(self.module(inst_batch)[0].detach().numpy())
        outputs = np.vstack(outputs)
        outputs = pd.DataFrame(outputs, index=X.index)
        outputs['pred_label'] = outputs.idxmax(axis=1)
        # change to return max labels...
        return outputs


    def load(self, path):
        model_details = torch.load(os.path.join(path, "model_detailed.pth"))

        self.topology = model_details["topology"]
        self.dropout = model_details["dropout"]
        self.L2Reg = model_details["L2Reg"]
        # note that we hardcode dropout to False as we do not want to have
        # dropout in testing (just in Training)
        self.module = fcnnClassifierModule(self.topology, dropout=0.0)
        self.module.load_state_dict(torch.load(os.path.join(path, "model.pth")))


class fcnnClassifierModule(nn.Module):
    def __init__(self, topology: list, dropout: float = 0.0):
        super().__init__()
        layers = np.array(topology).repeat(2)[1:-1]
        if dropout:
            nn_layers = np.array(
                [
                    [nn.Linear(int(a), int(b), bias=False), nn.ReLU(), nn.Dropout(dropout)]
                    for a, b in layers.reshape(-1, 2)
                ]
            ).flatten()[:-2]
        else:
            nn_layers = np.array(
                [
                    [nn.Linear(int(a), int(b), bias=False), nn.ReLU()]
                    for a, b in layers.reshape(-1, 2)
                ]
            ).flatten()[:-1]
        self._neural_net = IntermediateSequential(*nn_layers)

    def forward(self, inst_batch, return_act: bool = True):
        output, intermediate_outputs = self._neural_net(inst_batch)
        return output, intermediate_outputs

    def get_neural_net(self):
        return self._neural_net

    def erase_dropout(self):
        new_layers = []
        for layer in self.get_neural_net():
            if not isinstance(layer, nn.Dropout):
                new_layers.append(layer)
        self._neural_net = IntermediateSequential(*new_layers)


def reset_weights(layer):
    if isinstance(layer, nn.Linear):
        layer.reset_parameters()
