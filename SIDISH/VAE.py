from SIDISH.VAE_ARCHITECTURE import ARCHITECTURE as architecture
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import pyro
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn import metrics
import torch.optim as optim
import numbers
import os

class VAE():
    def __init__(self, epochs, adata, z_dim, layer_dims, optimizer, lr, dropout, device, seed):
        super(VAE, self).__init__()

        self.seed = seed

        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.backends.cudnn.deterministic = True
        np.random.seed(self.seed)
        os.environ["PYTHONHASHSEED"] = str(self.seed)

        ## Model parameters
        self.epochs = epochs
        self.imput_dim = adata.X.shape[1]
        self.lr = lr
        self.z_dim = z_dim
        self.layer_dims = layer_dims 
        self.dropout = dropout
        self.device = device

    def initialize(self, adata, W=None, batch_size=1024, type="Normal", num_workers=8):

        ## Initialise model
        self.adata = adata
        model = architecture(self.imput_dim, self.z_dim, self.layer_dims,self.seed, self.dropout)
        self.model = model.to(self.device)
        self.optimizer = optim.Adam(lr=self.lr, params=self.model.parameters())

        ## Initialise weights
        if W is None:
            W = np.ones(self.adata_train.X.shape)
        else:
            W = W
        self.W = W

        ## Get the cells by genes matrix X from the adata variable
        if type == 'Dense':
            data_list = [np.array(self.adata.X.todense()), self.W]
            data_list_total = [np.array(self.adata.X.todense()), self.W]
        else:
            data_list = [np.array(self.adata.X), self.W]
            data_list_total = [np.array(self.adata.X), self.W]

        ## Prep single cell data for training in VAE -- cell by genes matrix X with gene weight matrix W
        data_list = [torch.from_numpy(np.array(d)).type(torch.float) for d in data_list]
        dataset = TensorDataset(data_list[0].float(), data_list[1].float())
        kwargs = {'num_workers': num_workers, 'pin_memory':True}
        self.train_loader = DataLoader(dataset, batch_size=batch_size, **kwargs, drop_last=True)

        data_list_total = [torch.from_numpy(np.array(d)).type(torch.float) for d in data_list_total]
        total_dataset = TensorDataset(data_list_total[0].float(), data_list_total[1].float())
        kwargs = {'num_workers': num_workers, 'pin_memory':True}
        self.total_loader = DataLoader(total_dataset, batch_size=self.adata.X.shape[0], **kwargs)

        return self.model, self.train_loader

    def train(self):
        # training loop
        # here y is the weight
        self.loss = []

        for epoch in range(self.epochs):
            epoch_loss = 0.
            for x, y in self.train_loader:
                x = x.to(self.device, non_blocking=True)
                y_ = y.to(self.device, non_blocking=True)
                self.optimizer.zero_grad()
                mu_decoder, dropout_logits, mu_encoder, logvar = self.model(x)
                loss = self.model.loss_function(x, y_, mu_decoder, dropout_logits, mu_encoder, logvar)
                loss.backward()
                epoch_loss += loss.item() * x.size(0)
                self.optimizer.step()

            normalizer_train = len(self.train_loader.dataset)
            total_epoch_loss_train = epoch_loss / normalizer_train
            self.epoch_loss = total_epoch_loss_train
            self.loss.append(-total_epoch_loss_train)
            print("[epoch %03d]  average training loss: %.4f" %(epoch, self.epoch_loss))

    def getLoss(self): 
        self.error = np.array([i for i in self.loss])
        return self.error

    def getEmbedding(self, clustering=True):
        self.TZ = []
        self.model.eval()
        with torch.no_grad():
            for x, y in self.total_loader:
                # if on GPU put mini-batch into CUDA memory
                pyro.clear_param_store()
                x = x.to(self.device, non_blocking=True)
                z = self.model.get_latent_representation(x)
                zz = z.cpu().detach().numpy().tolist()
                self.TZ += zz

        if clustering:
            self.adata.obsm['latent'] = np.array(self.TZ).astype(np.float32)
        return self.adata

    def getARI(self):
        ARI = 0
        ARI = adjusted_rand_score(self.adata.obs['celltype_major'].tolist(), self.adata.obs['leiden_model'].tolist())
        return ARI

    def getSilhouette(self):
        data = np.array(self.TZ)
        labels = np.array(self.adata.obs['leiden_model'].values).astype(int)
        silhouette =  metrics.silhouette_score(data, labels)
        return silhouette

    def getBouldin(self):
        data = np.array(self.TZ)
        labels = np.array(self.adata.obs['leiden_model'].values).astype(int)
        bouldin =  metrics.davies_bouldin_score(data, labels)
        return bouldin

    def getNMI(self):
        NMI = 0
        NMI = normalized_mutual_info_score(self.adata.obs['celltype_major'].tolist(), self.adata.obs['leiden_model'].tolist())
        return NMI
