################################## IMPORT LIBRARY ############################
import os
import torch
import torch.nn as nn
import numpy as np

import pyro.contrib.examples.util
from pyro.distributions.zero_inflated import ZeroInflatedNegativeBinomial

import torch.nn.functional as F
from torch.distributions import kl_divergence as KL
from torch.distributions import Normal

from torch_geometric.nn import GCNConv
from torch_geometric.loader import NeighborSampler

# assert pyro.__version__.startswith('1.8.4')
pyro.distributions.enable_validation(False)

################################ MODEL ARCHITECTURE #########################################

class GraphConvLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphConvLayer, self).__init__()
        self.gcn = GCNConv(in_features, out_features)
        
    def forward(self, x, edge_index, edge_weight=None):
        """
        x: Node features [N, in_features]
        edge_index: Graph connectivity in COO format [2, E]
        edge_weight: Edge weights [E]
        """
        return self.gcn(x, edge_index, edge_weight)


class Decoder(nn.Module):
    def __init__(self, input_dim, z_dim, layer_dims, dropout=0):
        super(Decoder, self).__init__()
        layers = []
        in_dim = z_dim
        for dim in layer_dims:

            layers.append(nn.Linear(in_dim, dim))
            
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            else:
                None
                
            layers.append(nn.Softplus())
            #layers.append(nn.BatchNorm1d(dim))

            in_dim = dim

        self.before_last_layer = nn.Sequential(*layers)
        self.last_layer_1 = nn.Linear(in_dim, input_dim)
        self.last_layer_2 = nn.Linear(in_dim, input_dim)

    def forward(self, z):
        bll = self.before_last_layer(z)
        mu = self.last_layer_1(bll)
        dropout_logits = self.last_layer_2(bll)

        return torch.exp(mu), dropout_logits


class Encoder(nn.Module):
    def __init__(self, input_dim, z_dim, layer_dims, dropout=0):
        super(Encoder, self).__init__()
        layers = []
        in_dim = input_dim
        self.var_eps = 1e-4
        for dim in layer_dims:
            layers.append(nn.Linear(in_dim, dim))
            
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            else:
                None
                
            layers.append(nn.Softplus())
            #layers.append(nn.BatchNorm1d(dim))
         
            in_dim = dim

        self.before_last_layer = nn.Sequential(*layers)
        self.fc_mean = nn.Linear(in_dim, z_dim)
        self.fc_logvar = nn.Linear(in_dim, z_dim)

    def forward(self, x):
        hidden = self.before_last_layer(x)
        mean = self.fc_mean(hidden)
        logvar = self.fc_logvar(hidden) # + self.var_eps
        return mean, logvar

class SpatialPreEncoder(nn.Module):
    def __init__(self, input_dim, layer_dims, dropout=0):
        """
        Standard encoder - keeping the original implementation for Cox regression compatibility
        """
        super(SpatialPreEncoder, self).__init__()
        layers = []
        in_dim = input_dim
        self.var_eps = 1e-4
        for dim in layer_dims:
            layers.append(nn.Linear(in_dim, dim))
            
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            else:
                None
                
            layers.append(nn.Softplus())
            #layers.append(nn.BatchNorm1d(dim))
         
            in_dim = dim

        self.before_last_layer = nn.Sequential(*layers)

    def forward(self, x):
        hidden = self.before_last_layer(x)
        return hidden

class SpatialEncoder(nn.Module):
    def __init__(self, input_dim, z_dim, layer_dims, dropout=0, gcn_dims=None):
        """
        Spatial encoder that uses only the GCN-transformed features for latent space.
        """
        super(SpatialEncoder, self).__init__()

        # Create the standard encoder (MLP path)
        self.before_last_layer = SpatialPreEncoder(input_dim, layer_dims, dropout)
        self.fc_mean   = nn.Linear(layer_dims[-1], z_dim)
        self.fc_logvar = nn.Linear(layer_dims[-1], z_dim)

        # Use GCN path if provided
        self.use_spatial = gcn_dims is not None
        if self.use_spatial:
            last_hidden_dim = layer_dims[-1]
            self.gcn_layers = nn.ModuleList()
            gcn_in_dim = last_hidden_dim

            for gcn_dim in gcn_dims:
                self.gcn_layers.append(GraphConvLayer(gcn_in_dim, gcn_dim))
                gcn_in_dim = gcn_dim

            # Direct projection from GCN output to latent space
            self.gcn_mean = nn.Linear(gcn_dims[-1], z_dim)
            self.gcn_logvar = nn.Linear(gcn_dims[-1], z_dim)

    def forward(self, x, edge_index=None, edge_weight=None):
        hidden = self.before_last_layer(x)

        gcn_out = hidden
        for gcn_layer in self.gcn_layers:
            gcn_out = gcn_layer(gcn_out, edge_index, edge_weight)
            gcn_out = F.relu(gcn_out)

        mean = self.gcn_mean(gcn_out)
        logvar = self.gcn_logvar(gcn_out)

        return mean, logvar

    
class ARCHITECTURE(nn.Module):

    def __init__(self, input_dim, z_dim, layer_dims, seed, dropout=0.5, gcn_dims=None, use_cuda=False):
        super(ARCHITECTURE, self).__init__()

        self.seed = seed

        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.backends.cudnn.deterministic = True
        np.random.seed(self.seed)
        os.environ["PYTHONHASHSEED"] = str(self.seed)

        self.input_dim = input_dim
        self.z_dim = z_dim
        self.layer_dims = layer_dims
        self.use_cuda = use_cuda
        self.use_spatial = gcn_dims is not None

        self.log_theta = torch.nn.Parameter(torch.randn(self.input_dim))
        
        if self.use_spatial:
            self.encoder = SpatialEncoder(self.input_dim, self.z_dim, self.layer_dims, gcn_dims, dropout)
            self.decoder = Decoder(self.input_dim, self.z_dim, self.layer_dims[::-1], dropout)
            self.before_last_layer = self.encoder.before_last_layer
            
        else:
            # create the encoder and decoder networks
            self.encoder = Encoder(self.input_dim, self.z_dim, self.layer_dims, dropout)
            self.decoder = Decoder(self.input_dim, self.z_dim, self.layer_dims[::-1], dropout)
            self.before_last_layer = self.encoder.before_last_layer

    def reparameterize(self, mu, logvar):
        # Normal(mu, logvar).rsample()
        '''std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)'''
        return Normal(mu, logvar).rsample() #mu + eps * std

    def forward(self, x, edge_index=None, edge_weight=None):
        x = torch.log(x + 1)
        
        if self.use_spatial:
            mu_encoder, logvar = self.encoder(x.view(-1, self.input_dim), edge_index, edge_weight)
        
        else:
            mu_encoder, logvar = self.encoder(x.view(-1, self.input_dim))
            
        z = self.reparameterize(mu_encoder, logvar)

        # zinb distribution
        mu_decoder, dropout_logits = self.decoder(z)
        return mu_decoder, dropout_logits, mu_encoder, logvar

    def get_latent_representation(self, x, edge_index=None, edge_weight=None):
        x = torch.log(x + 1)
        if self.use_spatial:
            mu_encoder, logvar = self.encoder(x.view(-1, self.input_dim), edge_index, edge_weight)
        
        else:
            mu_encoder, logvar = self.encoder(x.view(-1, self.input_dim))
        
        return mu_encoder + torch.exp(0.5*logvar)
    
    
    def get_base_latent_representation(self, x):
        x = torch.log(x + 1)
        h  = self.before_last_layer(x.view(-1, self.input_dim))
        mu = self.base_mean(h)
        lv = self.base_logvar(h)
        return mu + torch.exp(0.5 * lv)

    def kl_d(self,mu, logvar):
        z_loc = torch.zeros_like(mu)
        z_scale = torch.ones_like(logvar)
        kl = KL(Normal(mu, logvar), Normal(z_loc, z_scale)).sum(dim=1)

        return kl #(-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1))

    def reconstruction_loss(self, x, mu, dropout_logits, w):
        '''
        x: input data
        mu: output of decoder
        dropout_logits: dropout logits of zinb distribution
        w: weights for each sample in x (same shape as x)
        '''
        theta = F.softplus(self.log_theta)
        nb_logits = (mu + 1e-5).log() - (theta + 1e-5).log()

        distribution = ZeroInflatedNegativeBinomial(total_count=theta, logits=nb_logits, gate_logits=dropout_logits, validate_args=False)

        log_prob = distribution.log_prob(x)

        log_prob = log_prob * w  # Apply the weights

        return -log_prob.sum(dim=-1)

    def loss_function(self, x,w, mu_decoder, dropout_logits, mu_encoder, logvar):
        reconstruction_loss = self.reconstruction_loss(x, mu_decoder, dropout_logits,w)
        kl_div = self.kl_d(mu_encoder, logvar)
        return torch.mean(reconstruction_loss, dim=0) + torch.mean(kl_div, dim=0)

######################################## END ############################################

