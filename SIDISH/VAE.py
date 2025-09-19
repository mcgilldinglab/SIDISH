from SIDISH.VAE_ARCHITECTURE import ARCHITECTURE as architecture
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import pyro
import torch.optim as optim
from torch_geometric.loader import NeighborLoader
from torch_geometric.data import Data
import os

class VAE():
    def __init__(self, epochs, adata, z_dim, layer_dims, lr, dropout, device, seed, gcn_dims=None):
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
        
        ## Spatial GCN parameters 
        if gcn_dims is not None:
            self.gcn_dims = gcn_dims
            self.use_spatial = True
            
        else:
            self.use_spatial = False

    def initialize(self, adata, W=None, batch_size=1024, type="Normal", num_workers=8, spatial_graph=None, num_neighbors=5):

        ## Initialise model
        self.adata = adata
        
        if self.use_spatial and spatial_graph is not None:
            # Spatial
            model = architecture(self.imput_dim, self.z_dim, self.layer_dims, self.seed, self.dropout, self.gcn_dims)
        
        else:
            # Non-spatial
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

        ## Setup mini-batching for the full dataset (prediction)
        data_list_total = [torch.from_numpy(np.array(d)).type(torch.float) for d in data_list_total]
        total_dataset = TensorDataset(data_list_total[0].float(), data_list_total[1].float())
        kwargs = {'num_workers': num_workers, 'pin_memory':True}
        self.total_loader = DataLoader(total_dataset, batch_size=self.adata.X.shape[0], **kwargs)


        ## Setup spatial graph and neighbor sampling
        self.use_neighbor_sampling = False
        if self.use_spatial and spatial_graph is not None:
            self.edge_index, self.edge_weight = spatial_graph
            
            if self.edge_weight is None:
                self.edge_weight = torch.ones(self.edge_index.size(1))
                
            # Create PyTorch Geometric Data object
            self.graph_data = Data( x=self.X_tensor, edge_index=self.edge_index, edge_attr=self.edge_weight, y=self.W_tensor)
            
            # Create neighbor sampler for mini-batch training
            self.use_neighbor_sampling = True
            self.train_loader_spatial = NeighborLoader(self.graph_data, num_neighbors=[num_neighbors], batch_size=batch_size, shuffle=True, num_workers=num_workers)

        if self.use_spatial:
            return self.model, self.train_loader_spatial
        else:
            return self.model, self.train_loader
        

    def train(self):
        # training loop
        # here y is the weight
        self.loss = []

        for epoch in range(self.epochs):
            epoch_loss = 0.
            samples_processed = 0
            
            # Choose training mode based on spatial data availability
            if self.use_spatial and self.use_neighbor_sampling:
                # Train with spatial neighborhood sampling
                for batch in self.train_loader_spatial:
                    batch = batch.to(self.device)
                    batch_size = batch.x.size(0)
                    
                    self.optimizer.zero_grad()
        
                    # Forward pass with sampled neighborhood
                    mu_decoder, dropout_logits, mu_encoder, logvar = self.model( batch.x, batch.edge_index, batch.edge_attr)
                    loss = self.model.loss_function(batch.x, batch.y, mu_decoder, dropout_logits, mu_encoder, logvar)
                    loss.backward()
                    epoch_loss += loss.item() * batch_size
                    samples_processed += batch_size
                    self.optimizer.step()
            
            else:
                # Standard training without spatial data
                for x, y in self.train_loader:
                    x = x.to(self.device, non_blocking=True)
                    y_ = y.to(self.device, non_blocking=True)
                    batch_size = x.size(0)
                    self.optimizer.zero_grad()
                    mu_decoder, dropout_logits, mu_encoder, logvar = self.model(x)
                    loss = self.model.loss_function(x, y_, mu_decoder, dropout_logits, mu_encoder, logvar)
                    loss.backward()
                    epoch_loss += loss.item() * batch_size
                    samples_processed += batch_size
                    self.optimizer.step()
                    
            # Calculate average loss for the epoch
            total_epoch_loss_train = epoch_loss / samples_processed
            self.epoch_loss = total_epoch_loss_train
            self.loss.append(-total_epoch_loss_train)
            print("[epoch %03d]  average training loss: %.4f" %(epoch, self.epoch_loss))

    def getBaseEncoder(self):
        return self.model.base_encoder

    def getLoss(self): 
        self.error = np.array([i for i in self.loss])
        return self.error

    def getEmbedding(self, clustering=True):
        """
        Get latent embeddings for the entire dataset
        """
        self.model.eval()
        
        if self.use_spatial and self.use_neighbor_sampling:
            # For spatial data, process in chunks to avoid OOM
            all_embeddings = []
            batch_size = 256  # Process in chunks
            num_cells = self.X_tensor.shape[0]
            
            for i in range(0, num_cells, batch_size):
                end_idx = min(i + batch_size, num_cells)
                batch_indices = list(range(i, end_idx))
                
                # Create subgraph for this batch and its neighbors
                node_idx = torch.tensor(batch_indices, dtype=torch.long)
                subgraph_loader = NeighborLoader(
                    self.graph_data, 
                    num_neighbors=[5],
                    batch_size=len(batch_indices),
                    input_nodes=node_idx,
                    shuffle=False,
                    num_workers=4
                )
                
                # Process each subgraph
                for batch in subgraph_loader:
                    batch = batch.to(self.device)
                    with torch.no_grad():
                        # Get embeddings
                        z = self.model.get_latent_representation(
                            batch.x, 
                            batch.edge_index, 
                            batch.edge_attr
                        )
                    
                    # Extract only the embeddings for the central nodes (not neighbors)
                    # The central nodes are always the first n nodes in the batch
                    central_node_embeddings = z[:len(batch_indices)].cpu().numpy()
                    all_embeddings.append(central_node_embeddings)
            
            # Combine all embeddings
            self.TZ = np.vstack(all_embeddings).tolist()
        else:
            # For non-spatial data, use the standard approach
            self.TZ = []
            with torch.no_grad():
                for x, y in self.total_loader_nonspatial:
                    pyro.clear_param_store()
                    x = x.to(self.device, non_blocking=True)
                    z = self.model.get_latent_representation(x)
                    zz = z.cpu().detach().numpy().tolist()
                    self.TZ += zz

        if clustering:
            self.adata.obsm['latent'] = np.array(self.TZ).astype(np.float32)
        return self.adata
    
    def getBaseEmbedding(self, clustering=True):
        """
        Get embeddings using only the base encoder (for Cox regression compatibility)
        """
        self.base_TZ = []
        self.model.eval()
        with torch.no_grad():
            for x, y in self.total_loader_nonspatial:
                pyro.clear_param_store()
                x = x.to(self.device, non_blocking=True)
                
                # Get latent representation with base encoder only (no spatial)
                z = self.model.get_base_latent_representation(x)
                
                zz = z.cpu().detach().numpy().tolist()
                self.base_TZ += zz

        if clustering:
            self.adata.obsm['base_latent'] = np.array(self.base_TZ).astype(np.float32)
        return self.adata

