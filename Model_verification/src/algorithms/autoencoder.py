import os
import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd

from .neural_net import neural_net
from .neural_net import smallest_k_dist_loss
from .neural_net import IntermediateSequential


class autoencoder(neural_net):
    def __init__(
        self,
        topology: list = [2,1,2],
        name: str = "autoencoder",
        bias: bool = True,
        fct_dist: list = [],
        lambda_fct: float = 0.1,
        robust_ae: float = None,
        denoising: float = None,
        border_dist: bool = False,
        lambda_border: float = 0.01,
        penal_dist: float = 0.01,
        num_border_points: int = 1,
        dropout: bool = False,
        L2Reg: float = 0,
        num_epochs: int = 100,
        batch_size: int = 20,
        lr: float = 1e-3,
        seed: int = None,
    ):
        super().__init__(
            name, num_epochs, batch_size, lr, seed
        )
        self.border_dist = border_dist
        self.fct_dist = fct_dist
        self.robust_ae = robust_ae
        self.denoising = denoising
        self.topology = topology
        self.lambda_border = lambda_border
        self.penal_dist = penal_dist
        self.lambda_fct = lambda_fct
        self.bias = bias
        self.dropout = dropout
        self.L2Reg = L2Reg
        self.num_border_points = num_border_points
        self.module = autoencoderModule(self.topology, self.bias, self.dropout)

    def fit(self, X: pd.DataFrame, run_folder=None, logger=None):
        # check when this is needed...
        data_loader = self.get_data_loader(X)

        # incorporates L2 reg
        optimizer = torch.optim.Adam(
            params=self.module.parameters(), lr=self.lr, weight_decay=self.L2Reg
        )

        for epoch in range(self.num_epochs):
            epoch_start = datetime.datetime.now()
            self.module.train()
            epoch_recons_loss = 0
            epoch_reg_loss = 0
            for inst_batch in data_loader:

                # reconstr loss with or without denoising
                if self.denoising:
                    noise = np.random.normal(
                        0, scale=self.denoising, size=inst_batch.shape
                    ).astype(np.float32)
                    inst_batch_noise = inst_batch + noise
                    reconstr = self.module(inst_batch_noise)[0]
                else:
                    reconstr = self.module(inst_batch)[0]
                loss_reconstr = nn.MSELoss()(inst_batch, reconstr)
                epoch_recons_loss += loss_reconstr

                # fctdist and borderdist block
                loss_border, loss_fct = smallest_k_dist_loss(
                    self.num_border_points,
                    border_dist=self.border_dist,
                    penal_dist=self.penal_dist,
                    fct_dist=self.fct_dist,
                )(inst_batch, self.module)
                epoch_reg_loss += loss_border + loss_fct

                # total loss + gradient step
                loss = (
                    loss_reconstr
                    + self.lambda_border * loss_border
                    + self.lambda_fct * loss_fct
                )
                self.module.zero_grad()
                loss.backward()
                optimizer.step()


            # robust_ae block
            self.module.eval()
            if self.robust_ae is not None:
                if epoch == 0:
                    L_D = self.predict(X)
                else:
                    L_D = self.predict(L_D)
                S = X - L_D
                S = self.prox_l1(S)
                L_D = X - S
                data_loader = self.get_data_loader(L_D)

            epoch_end = datetime.datetime.now()
            duration = epoch_end - epoch_start

            if run_folder:
                self.save(run_folder)
            if logger:
                logger.info(f"Training epoch {epoch}")
                logger.info(f"Epoch_loss: {epoch_recons_loss + epoch_reg_loss}")
                logger.info(f"Epoch_loss_reconstr: {epoch_recons_loss}")
                logger.info(f"Epoch_loss_regularization: {epoch_reg_loss}")
                logger.info(f"Duration: {duration}")

        self.module.erase_dropout()


    def prox_l1(self, S):
        S[abs(S) < self.robust_ae] = 0
        S[S > self.robust_ae] -= self.robust_ae
        S[S < self.robust_ae] += self.robust_ae
        return S

    def predict(self, X: pd.DataFrame):
        self.module.eval()
        data_loader = DataLoader(
            dataset=X.values,
            batch_size=self.batch_size,
            drop_last=False,
            pin_memory=True,
        )
        reconstructions = []
        # TODO check if possible to do with forward
        for inst_batch in data_loader:
            inst_batch = inst_batch.float()
            reconstructions.append(self.module(inst_batch)[0].detach().numpy())
        reconstructions = np.vstack(reconstructions)
        reconstructions = pd.DataFrame(reconstructions, index=X.index)
        return reconstructions

    def extract_latent(self, X: pd.DataFrame):
        self.module.eval()
        data_loader = DataLoader(
            dataset=X.values,
            batch_size=self.batch_size,
            drop_last=False,
            pin_memory=True,
        )
        latent_repr = []
        latent_layer_key = str(int(len(self.module.get_neural_net()) / 2))
        # TODO same as in predict (use forward and extract intermed. result)
        for inst_batch in data_loader:
            inst_batch = inst_batch.float()
            latent_repr.append(
                self.module(inst_batch)[1][latent_layer_key].detach().numpy()
            )
        latent_repr = np.vstack(latent_repr)
        latent_repr = pd.DataFrame(latent_repr)
        return latent_repr

    def load(self, path):
        model_details = torch.load(os.path.join(path, "model_detailed.pth"))

        self.topology = model_details["topology"]
        self.denoising = model_details["denoising"]
        self.fct_dist = model_details["fct_dist"]
        self.border_dist = model_details["border_dist"]
        self.num_border_points = model_details["num_border_points"]
        self.bias = model_details["bias"]
        self.dropout = model_details["dropout"]
        self.L2Reg = model_details["L2Reg"]
        self.lambda_border = model_details["lambda_border"]
        self.lambda_fct = model_details["lambda_fct"]
        # note that we hardcode dropout to False as we want to ensure to not
        # have dropout in testing
        self.module = autoencoderModule(self.topology, self.bias, dropout=False)
        self.module.load_state_dict(torch.load(os.path.join(path, "model.pth")))

class autoencoderModule(nn.Module):
    def __init__(self, topology: list, bias: bool = True, dropout: bool = False):
        super().__init__()
        layers = np.array(topology).repeat(2)[1:-1]
        if len(topology) % 2 == 0:
            print(layers)
            raise Warning(
                """Your topology is probably not what you want as the
                    hidden layer is repeated multiple times"""
            )
        if dropout:
            nn_layers = np.array(
                [
                    [nn.Linear(int(a), int(b), bias=bias), nn.ReLU(), nn.Dropout(p=0.1)]
                    for a, b in layers.reshape(-1, 2)
                ]
            ).flatten()[:-2]
        else:
            nn_layers = np.array(
                [
                    [nn.Linear(int(a), int(b), bias=bias), nn.ReLU()]
                    for a, b in layers.reshape(-1, 2)
                ]
            ).flatten()[:-1]
        self._neural_net = IntermediateSequential(*nn_layers)

    def forward(self, inst_batch, return_act: bool = True):
        reconstruction, intermediate_outputs = self._neural_net(inst_batch)
        return reconstruction, intermediate_outputs

    def get_neural_net(self):
        return self._neural_net

    def erase_dropout(self):
        new_layers = []
        for layer in self.get_neural_net():
            if not isinstance(layer, nn.Dropout):
                new_layers.append(layer)
        self._neural_net = IntermediateSequential(*new_layers)
