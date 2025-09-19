
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
from tqdm import tqdm
from lifelines.utils import concordance_index
import os
from SIDISH.DEEP_COX_ARCHITECTURE import DEEPCOX_ARCHITECTURE as deepCox


def loss_DeepCox(pred, events, durations, weight=None, train=True):
    """
        Compute the negative log-likelihood for the Deep Cox model in Phase 2 of SIDISH.

        Parameters
        ----------
        pred : torch.Tensor
            Predicted risk scores.
        events : torch.Tensor
            Event indicators (1 if event occurred, 0 otherwise or censored).
        durations : torch.Tensor
            Time durations.
        weight : torch.Tensor
            Patient weights.
        train : bool, optional
            Whether the model is in training mode. Defaults to True.

        Returns
        -------
        torch.Tensor
            Negative log-likelihood.

        Notes
        -----
        This method is based on the implementation from DeepSurv:
        https://github.com/jaredleekatzman/DeepSurv/blob/master/deepsurv/deep_surv.py
    """
    if train:
        idx = durations.sort(descending=True)[1] # Sort by durations in descending order
        events = events[idx] # Sort events by durations
        pred = pred[idx] # Sort risk predictions by durations
        weight = weight[idx] # Sort patient weight by durations

        hazard_ratio = torch.exp(pred) / weight
        log_risk = torch.log(torch.cumsum(hazard_ratio, dim=0))

        uncensored_likelihood = pred.t() - log_risk
        censored_likelihood = uncensored_likelihood * events
        num_observed_events = torch.sum(events)
        neg_likelihood = -torch.sum(censored_likelihood) / num_observed_events

    elif train == False:
        idx = durations.sort(descending=True)[1]
        events = events[idx]
        pred = pred[idx]

        hazard_ratio = torch.exp(pred)
        log_risk = torch.log(torch.cumsum(hazard_ratio, dim=0))

        uncensored_likelihood = pred.t() - log_risk
        censored_likelihood = uncensored_likelihood * events
        num_observed_events = torch.sum(events)
        neg_likelihood = -torch.sum(censored_likelihood) / num_observed_events

    return neg_likelihood


class DEEPCOX():

    def __init__(self, X_train, Y_train, weights, hidden, encoder, device, batch_size,seed, lr=0.000001, dropout=0):

        self.device = device
        self.X_train = X_train
        self.Y_train = Y_train
        self.weights = weights

        self.hidden = hidden
        self.encoder = encoder 
        self.dropout = dropout
        self.lr = lr
        self.seed = seed

        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.backends.cudnn.deterministic = True
        np.random.seed(self.seed)
        os.environ["PYTHONHASHSEED"] = str(self.seed)

        # Initializing the Deep Cox regression model of phase 2 of SIDISH
        self.model = deepCox(hidden=self.hidden, encoder=self.encoder, dropout=self.dropout)

        # Transfer Learning, unfrezzing the weights of the encoder from the VAE from phase 1. Continuing the training from where it left off.
        for name, para in self.model.encoder_layer.named_parameters():
            para.requires_grad = True
        self.non_frozen_parameters = [p for p in self.model.parameters() if p.requires_grad]
        self.model = self.model.to(self.device)
        self.opt = optim.Adam(self.non_frozen_parameters, lr=self.lr)

        self.X_train[:, -1] = self.weights
        train_dataset = TensorDataset(self.X_train.float(), self.Y_train.float())
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    def train(self, epochs):

        # Training the Deep Cox Regression model
        for epoch in tqdm(range(epochs)):
            for x, y in self.train_loader:
                self.model.train()
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)

                x_ = x[:, :-1]
                w = x[:, -1:].flatten()
                days = y[:, :-1].flatten()
                events = y[:, -1:].flatten()

                pred = self.model(x_).flatten()
               
                loss = loss_DeepCox(pred, events, days, w).view(1,)

                self.opt.zero_grad(set_to_none=True)
                loss.backward()  
                self.opt.step()

        # Loss and Concordance index calclation
        out = -np.exp(pred.detach().cpu().numpy().reshape((1, -1))[0])
        self.ci_train = concordance_index(days.cpu().numpy(), out, events.cpu().numpy()) 
        self.loss_train = loss.item()

    def get_train_ci(self):
        return self.ci_train

    def get_train_loss(self):
        return self.loss_train

    def get_test_loss(self, test_loader):
        self.model.eval()
        with torch.no_grad():
            genes, surv = next(iter(test_loader))
            genes = genes.to(self.device, non_blocking=True)
            surv = surv.to(self.device, non_blocking=True)

            genes_ = genes[:, :-1]
            w_test = genes[:, -1:]
            days_test = surv[:, :-1].flatten()
            events_test = surv[:, -1:].flatten()

            pred_test = self.model(genes_).flatten()
            loss_test = loss_DeepCox(pred_test, events_test, days_test, train=False).view(1,)
        return loss_test

    def get_test_ci(self, test_loader):
        self.model.eval()
        with torch.no_grad():
            genes, surv = next(iter(test_loader))
            genes = genes.to(self.device, non_blocking=True)
            surv = surv.to(self.device, non_blocking=True)

            genes_ = genes[:, :-1]
            w_test = genes[:, -1:]
            days_test = surv[:, :-1].flatten()
            events_test = surv[:, -1:].flatten()

            pred_test = self.model(genes_).flatten()
            out = -np.exp(pred_test.detach().cpu().numpy().reshape((1, -1))[0])
            ci_test = concordance_index(days_test.cpu().numpy(), out, events_test.cpu().numpy())
        return ci_test
    
    

