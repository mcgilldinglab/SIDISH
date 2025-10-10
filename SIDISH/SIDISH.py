# Core SIDISH components
from SIDISH.DEEP_COX import DEEPCOX as DeepCox          # Deep Cox model wrapper (Phase 2)
from SIDISH.VAE import VAE as VAE                        # VAE encoder (Phase 1)
from SIDISH.Utils import Utils as utils                  # Helper utilities used across phases
from SIDISH.Utils import get_spatial_graph_from_adata    # Build spatial graph (for spatial-VAE)
from SIDISH.in_silico_perturbation import InSilicoPerturbation
from SIDISH.ppi_network_handler import PPINetworkHandler # Loads/queries PPI network
from SIDISH.gene_perturbation_utils import GenePerturbationUtils  # Gene KO / network-based adj.


from statsmodels.stats.multitest import multipletests
import seaborn as sns
import pandas as pd
import numpy as np
import torch
import scanpy as sc
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import os
from tqdm import tqdm

from typing import Literal
from lifelines import CoxPHFitter
from lifelines.statistics import logrank_test
from lifelines import KaplanMeierFitter
import math
import itertools
from scipy.stats import binomtest
from scipy.stats import wilcoxon
import pyro

def process_Data(X: np.ndarray, Y: np.ndarray, test_size: float, batch_size: int, seed: int) -> tuple:

    """
    Splits bulk RNA-seq data into training and testing datasets, converts them to tensors, 
    and creates DataLoaders.

    Parameters
    ----------
    X : np.ndarray
        Bulk gene expression data.
    Y : np.ndarray
        Survival data: [survival days, event, weight].
    test_size : float
        Proportion of dataset allocated to the test split.
    batch_size : int
        Number of patients per batch.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    tuple
        - torch.Tensor: X_train (Training feature matrix)
        - torch.Tensor: X_test (Testing feature matrix)
        - torch.Tensor: y_train (Training labels)
        - torch.Tensor: y_test (Testing labels)
    """

    # Split data into train, val, test dataset
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed, stratify=Y[:, 1])

    # Turn to tensor fromat
    data_list = [X_train, y_train, X_test, y_test]
    data_list = [torch.from_numpy(np.array(d)).type(torch.float) for d in data_list]

    X_train = data_list[0]
    y_train = data_list[1]
    X_test = data_list[2]
    y_test = data_list[3]

    # Train_dataset - X_train, y_train
    train_dataset = TensorDataset(data_list[0].float(), data_list[1].float())

    # Test_dataset - X_test, y_test
    test_dataset = TensorDataset(data_list[2].float(), data_list[3].float())

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=X_test.shape[0], shuffle=False)

    return X_train, X_test, y_train, y_test

def plot_umap(ax, umap_combined, palette, percentage_change_):
    """
    Scatter UMAP of High-Risk/Background status after perturbation with a 
    custom palette and an extra legend line indicating the perturbation percentage change.
    """
    sns.scatterplot(
        x="UMAP1", 
        y="UMAP2", 
        hue='status', 
        data=umap_combined, 
        palette=palette, 
        edgecolor='none',
        alpha=1, 
        s=15, 
        ax=ax
    )
    
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlabel("UMAP1", fontsize=12)
    ax.set_ylabel("UMAP2", fontsize=12)
    
    ax.set_xticks([])
    ax.set_yticks([])
    # Update legend to include perturbation score
    handles, labels = ax.get_legend_handles_labels()
    handles.append(plt.Line2D([0], [0]))
    labels.append(f"{percentage_change_:.2f}%")
    ax.legend(handles, labels, loc="upper left", fontsize=12, bbox_to_anchor=(0.95, 0.75), frameon=False)


def plot_umap_differential(ax, umap_combined):
    """
    Scatter UMAP colored by continuous risk delta after perturbation.
    """
    sns.scatterplot(
        x="UMAP1", 
        y="UMAP2", 
        hue='risk', 
        data=umap_combined, 
        palette="rocket", 
        edgecolor='none',
        alpha=1, 
        s=15, 
        ax=ax,
        legend=False 
    )
    
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlabel("UMAP1", fontsize=12)
    ax.set_ylabel("UMAP2", fontsize=12)
    
    ax.set_xticks([])
    ax.set_yticks([])
    # Update legend to include perturbation score
    handles, labels = ax.get_legend_handles_labels()
    handles.append(plt.Line2D([0], [0]))
    ax.legend(handles, labels, loc="upper left", fontsize=12, bbox_to_anchor=(0.95, 0.75), frameon=False)


def preprocess(
    adata, 
    bulk, 
    survival_df, 
    patient_id, 
    celltype_name, 
    processed = True, 
    n_genes_by_counts = 5000, 
    pct_counts_mt = 10, 
    batch_correction=False, 
    survival_ = "Overall_survival_days",  
    status = "Sample_Status"
):
    
    """
    Harmonize scRNA-seq (AnnData) and bulk tables:
    - QC (if raw), HVG selection, intersection of genes across modalities
    - Optionally apply Harmony (neighbors/umap/visual check) + ComBat
    - Merge survival metadata -> bulk with columns: duration, event

    Returns
    -------
    (AnnData, pd.DataFrame)
        (sc object restricted to intersecting genes, bulk with survival columns)
    """
    subset = None 

    ## Single-cell data preprocessing
    if processed == False:
        adata.var_names_make_unique()  
        sc.pp.filter_cells(adata, min_genes=3)
        sc.pp.filter_genes(adata, min_cells=400)
        
        # annotate mito genes
        adata.var['mt'] = adata.var_names.str.startswith(('MT-','mt-'))
        sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)

        # QC plots
        sc.pl.violin(adata, ['n_genes_by_counts', 'total_counts', 'pct_counts_mt'], jitter=0.4, multi_panel=True)
        sc.pl.scatter(adata, x='total_counts', y='pct_counts_mt')
        sc.pl.scatter(adata, x='total_counts', y='n_genes_by_counts')
        
        # cell-level thresholds
        adata = adata[adata.obs.n_genes_by_counts < n_genes_by_counts, :].copy()
        adata = adata[adata.obs.pct_counts_mt < pct_counts_mt, :].copy()
        
        # normalize/log + HVG
        adata.raw = adata.copy()
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        sc.pp.highly_variable_genes(adata, flavor='seurat')

        # count cells restricted to HVGs
        subset = adata.raw.to_adata()
        subset = subset[:, adata.var.highly_variable].copy()
        adata = adata[:, adata.var.highly_variable].copy()
        
        data = bulk.copy()  # assumes first two columns are metadata (survival days, event)
        bulk_genes = data.columns.to_numpy()
        
        if 'gene_ids' in adata.var.columns: sc_gene = adata.var['gene_ids'].astype(str).values
        else: sc_gene = adata.var_names.astype(str).values

        inter = np.intersect1d(bulk_genes, sc_gene)
        if inter.size == 0:
            raise ValueError("No overlapping genes between bulk and scRNA-seq.")

        ## Keeping intersection between scRNA-seq and bulk
        if 'gene_ids' in adata.var.columns:
            adata = adata[:, adata.var['gene_ids'].astype(str).isin(inter)].copy()
            subset = subset[:, subset.var['gene_ids'].astype(str).isin(inter)].copy()
        else:
            adata = adata[:, adata.var_names.isin(inter)].copy()
            subset = subset[:, subset.var_names.isin(inter)].copy()
        if 'cells' in subset.obs.columns:
            subset.obs_names = subset.obs['cells'].astype(str).values
        if celltype_name in subset.obs.columns:
            subset.obs.rename(columns={celltype_name: "celltype_major"}, inplace=True)
        
        data = data.filter(items=adata.to_df().columns.values)
        data = data[adata.to_df().columns.values]
        
        bulk = pd.concat([survival_df, data], axis=1)
        bulk.rename(columns={survival_: "duration", status: "event"}, inplace=True)
        change = [1 if str(i).strip().lower() == "dead" else 0 for i in bulk["event"].values]
        bulk["event"] = change

        ## Optional UMAP and batch correction using Harmony
        if batch_correction:
            sc.pp.scale(adata, max_value=10)
            sc.tl.pca(adata, svd_solver='arpack')
            sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)
            sc.external.pp.harmony_integrate(adata, key=patient_id)
            sc.tl.umap(adata)
            sc.pl.umap(adata, color=[patient_id])
        
            ## Data batch correction using ComBat 
            if patient_id not in subset.obs.columns:
                subset.obs[patient_id] = subset.obs_names.str.split('_').str[-1]
            sc.pp.combat(subset, key=patient_id)
        return subset, bulk

    else:
        # processed == True
        if 'cells' in adata.obs.columns:
            adata.obs_names = adata.obs['cells'].astype(str).values
        if celltype_name in adata.obs.columns:
            adata.obs.rename(columns={celltype_name: "celltype_major"}, inplace=True)
            
        data = bulk.copy()
        bulk_genes = data.columns.to_numpy()

        # Restrict sc to intersection
        if 'gene_ids' in adata.var.columns: sc_gene = adata.var['gene_ids'].astype(str).values
        else: sc_gene = adata.var_names.astype(str).values
        
        inter = np.intersect1d(bulk_genes, sc_gene)

        if inter.size == 0:
            raise ValueError("No overlapping genes between bulk and scRNA-seq.")
        
        ## Keeping intersection between scRNA-seq and bulk
        if 'gene_ids' in adata.var.columns:
            adata = adata[:, adata.var['gene_ids'].astype(str).isin(inter)].copy()
        else:
            adata = adata[:, adata.var_names.isin(inter)].copy()
        
        # Reorder bulk to sc gene order
        data = data.filter(items=adata.to_df().columns.values)
        data = data[adata.to_df().columns.values]
        
        # Merge survival metadata; map event to {0,1}
        bulk = pd.concat([survival_df, data], axis=1)
        bulk.rename(columns={survival_: "duration", status: "event"}, inplace=True)
        change = [1 if str(i).strip().lower() == "dead" else 0 for i in bulk["event"].values]
        bulk["event"] = change
        
        return adata, bulk


class SIDISH:
    """
    SIDISH (Semi-Supervised Iterative Deep Learning for Identifying High-Risk Cells).

    This framework integrates single-cell and bulk RNA-seq data to identify 
    High-Risk cancer cells and potential biomarkers.

    Parameters
    ----------
    adata : AnnData
        Single-cell RNA-seq data.
    bulk : pd.DataFrame
        Bulk RNA-seq data.
        
    use_spatial_graph : bool, optional
        Whether to use spatial graph information (default=False).

    k_neighbors : int, optional
        Number of neighbors to use for constructing the spatial graph (default=5).
        
    device : str
        Computation device ('cpu' or 'cuda').
    seed : int, optional
        Random seed for reproducibility (default=1234).
    """

    def __init__(self, adata, bulk, device: str = "cpu", seed: int = 1234, use_spatial_graph: bool = False, k_neighbors: int = None) -> None:
        self.adata = adata
        self.bulk = bulk
        self.device = device
        self.seed = seed
        self.use_spatial_graph = use_spatial_graph
        self.k_neighbors = k_neighbors
        
        if self.use_spatial_graph and "spatial" in self.adata.obsm:
            edge_index, edge_weight = get_spatial_graph_from_adata(self.adata, spatial_key="spatial", method="knn", k=self.k_neighbors)
            self.spatial_graph = (edge_index, edge_weight)
            print("SIDISH Spatial graph constructed using k-NN with k =", self.k_neighbors)
            
        else:
            print("SIDISH No spatial graph used. Proceeding with dense VAE.")

    def init_Phase1(self, epochs: int, i_epochs: int, latent_size: int, layer_dims: list, batch_size: int, optimizer: str, lr: float, lr_3: float, dropout: float, type: str = 'Normal') -> None:
        """
        Initializes Phase 1: training a Variational Autoencoder (VAE) on single-cell RNA-seq data.
        
        Parameters
        ----------
        epochs : int
            Number of epochs for initial VAE training.
        i_epochs : int
            Number of iterations for retraining VAE.
        latent_size : int
            Latent dimension size.
        layer_dims : list
            List of hidden layer dimensions.
        batch_size : int
            Batch size.
        optimizer : str
            Optimizer for VAE training.
        lr : float
            Learning rate.
        lr_3 : float
            Learning rate for later iterations.
        dropout : float
            Dropout rate.
        type : str, optional
            Specifies dense or normal representation (default="Normal").

        Returns
        -------
        None
        """

        self.epochs_1 = epochs
        self.epochs_3 = i_epochs
        self.latent_size = latent_size
        self.layer_dims = layer_dims
        self.optimizer = optimizer
        self.lr_1 = lr
        self.lr_3 = lr_3
        self.dropout_1 = dropout
        self.batch_size = batch_size
        self.type = type

        # Initialise the weight matrix of phase 1
        self.W_matrix = np.ones(self.adata.X.shape)

    def init_Phase2(self, epochs: int, hidden: int, lr: float, dropout: float, test_size: float, batch_size_bulk: int) -> None:
        """
        Initializes Phase 2: training a Deep Cox model for survival analysis using bulk RNA-seq data.
        
        Parameters
        ----------
        epochs : int
            Number of training epochs for Deep Cox model.
        hidden : int
            Number of neurons in the hidden layer.
        lr : float
            Learning rate for Deep Cox model.
        dropout : float
            Dropout rate for training.
        test_size : float
            Proportion of dataset allocated to the test split.
        batch_size_bulk : int
            Number of samples per batch for bulk data.

        Returns
        -------
        None
        """
        self.epochs_2 = epochs
        self.hidden = hidden
        # self.iterations = iterations
        self.lr_2 = lr
        self.dropout_2 = dropout
        self.batch_size_bulk = batch_size_bulk

        # Initialise the weight  vector of phase 2
        self.W_vector = np.ones(self.bulk.iloc[:,2:].shape[0])
        self.bulk['weight'] = self.W_vector

        self.X = self.bulk.iloc[:, 2:].values
        self.Y = self.bulk.iloc[:, :2].values
        self.X_train, self.X_test, self.y_train, self.y_test = process_Data(self.X, self.Y, test_size, batch_size_bulk, self.seed)

    def train(self, iterations: int, percentile: float, steepness: float, path: str, num_workers: int = 0, show: bool = True, distribution_fit:Literal["default", "fitted"] = "default") -> sc.AnnData:
        """
        Trains the SIDISH framework iteratively, refining the identification of High-Risk cells.

        This function iteratively updates High-Risk cell classifications by integrating 
        single-cell and bulk RNA-seq data. Each iteration includes:
        - Training the VAE model on single-cell data.
        - Training the Deep Cox model on bulk RNA-seq survival data.
        - Updating weight matrices to improve High-Risk cell identification.

        Parameters
        ----------
        iterations : int
            Number of training iterations.
        percentile : float
            Threshold percentile for defining High-Risk cells.
        steepness : float
            Scaling factor for updating weights.
        path : str
            Directory for saving model checkpoints.
        num_workers : int, optional
            Number of parallel workers (default=8).
        show : bool, optional
            If True, displays training progress (default=True).

        Returns
        -------
        sc.AnnData
            Updated AnnData object containing the refined High-Risk cell classifications.
        """
        os.makedirs(path, exist_ok=True)
        self.path = path
        self.num_workers = num_workers
        self.percentile = percentile
        self.steepness = steepness

        # Re-Initialise the weight  vector of phase 2
        test_dataset = TensorDataset(self.X_test.float(), self.y_test.float())
        self.test_loader = DataLoader(test_dataset, batch_size=self.X_test.shape[0], shuffle=False)
        self.W_vector = self.X_train[:, -1]

        # Initialise the VAE of phase 1
        if self.use_spatial_graph and self.spatial_graph is not None:
            print("########################################## Using Spatial Graph in VAE ##########################################")
            self.vae = VAE(epochs=self.epochs_1, adata=self.adata, z_dim=self.latent_size, layer_dims=self.layer_dims, lr=self.lr_1, dropout=self.dropout_1, device=self.device, seed=self.seed, gcn_dims=[32, self.latent_size]) 
            self.vae.initialize(self.adata, W=self.W_matrix, batch_size=self.batch_size, num_workers=self.num_workers, spatial_graph=self.spatial_graph, num_neighbors=self.k_neighbors)
        
        else:
            print("########################################## Using Dense VAE ##########################################")
            self.vae = VAE(self.epochs_1,self.adata,self.latent_size, self.layer_dims, self.lr_1, self.dropout_1,self.device, self.seed)
            self.vae.initialize(self.adata, self.W_matrix, self.batch_size, self.type, self.num_workers)

        # Initial training of VAE in iteration 1
        print("########################################## ITERATION 1 OUT OF {} ##########################################".format(iterations))
        self.vae.train()

        # Save VAE for iterative process
        torch.save(self.vae.model.state_dict(), "{}vae_transfer".format(self.path))

        self.train_loss_Cox = []
        self.train_ci_Cox = []
        self.test_loss_Cox = []
        self.test_ci_Cox = []
        self.percentile_list = []
        for i in range(iterations):
            self.encoder = self.vae

            self.deepCox_model = DeepCox(self.X_train, self.y_train, self.W_vector, self.hidden, self.encoder, self.device,self.batch_size,self.seed, self.lr_2, self.dropout_2)
            self.deepCox_model.train(self.epochs_2)

            self.train_loss_Cox.append(self.deepCox_model.get_train_loss())
            self.train_ci_Cox.append(self.deepCox_model.get_train_ci())
            self.test_loss_Cox.append(self.deepCox_model.get_test_loss(self.test_loader))
            self.test_ci_Cox.append(self.deepCox_model.get_test_ci(test_loader=self.test_loader))

            patients_data = self.X_train[:, :-1].to(self.device)
            
            print("########################################## Calculating Patients Weight Vector ##########################################")
            if i == 0:
                self.scores, self.adata_, self.percentile_cells, self.cells_max, self.cells_min, returned_dist = utils.getWeightVector(patients_data, self.vae.adata, self.deepCox_model.model, self.percentile, self.device, distribution_fit)
                self.dist = returned_dist
            else: 
                if distribution_fit == 'fitted':
                    self.scores, self.adata_, self.percentile_cells, self.cells_max, self.cells_min , returned_dist = utils.getWeightVector(patients_data, self.vae.adata, self.deepCox_model.model, self.percentile, self.device, distribution_fit, self.dist)
                else:
                    self.scores, self.adata_, self.percentile_cells, self.cells_max, self.cells_min , returned_dist = utils.getWeightVector(patients_data, self.vae.adata, self.deepCox_model.model, self.percentile, self.device, "default")
            
                
            self.percentile_list.append(self.percentile_cells)
            self.W_vector += self.scores
            
            print("########################################## Calculating Cells Weight Matrix ##########################################")
            self.W_temp = utils.getWeightMatrix(self.adata_, self.seed, self.steepness, self.type)
            self.W_matrix += self.W_temp
            self.W_matrix[self.W_matrix >= 2] = 2

            self.adata = self.adata_.copy()

            print("########################################## Saving Weight Matrix at Iteration {} ##########################################".format(i))
            pd.DataFrame(self.W_matrix).to_csv("{}W_matrix_{}.csv".format(self.path,i))
            
            if i == (iterations - 1):
                print("########################################## SIDISH TRAINING DONE ##########################################")
                break

            else:
                print("########################################## ITERATION {} OUT OF {} ##########################################".format(i+2, iterations))
                if self.use_spatial_graph and self.spatial_graph is not None:
                    self.vae = VAE(epochs=self.epochs_3, adata=self.adata, z_dim=self.latent_size, layer_dims=self.layer_dims, lr=self.lr_3, dropout=self.dropout_1, device=self.device, seed=self.seed, gcn_dims=[32, self.latent_size])
                    self.vae.initialize(self.adata, spatial_graph=self.spatial_graph, W=self.W_matrix, batch_size=self.batch_size, num_neighbors=self.k_neighbors, num_workers=self.num_workers)
                    self.vae.model.load_state_dict(torch.load("{}vae_transfer".format(self.path)))
                
                else:
                    self.vae = VAE(self.epochs_3,self.adata,self.latent_size, self.layer_dims,self.lr_3, self.dropout_1,self.device, self.seed)
                    self.vae.initialize(self.adata, self.W_matrix, self.batch_size, self.type,self.num_workers)
                    self.vae.model.load_state_dict(torch.load("{}vae_transfer".format(self.path)))
                
                self.vae.train()

                # Save VAE for iterative process
                torch.save(self.vae.model.state_dict(), "{}vae_transfer".format(self.path))

        torch.save(self.deepCox_model.model.state_dict(), "{}deepCox".format(self.path))

        print("########################################## Saving Final AnnData Object ##########################################")
        fn = "{}adata_SIDISH.h5ad".format(self.path)
        self.adata.write_h5ad(fn, compression="gzip")
        return self.adata

    
    def getEmbedding_adata(self) -> sc.AnnData:
        """
        Extracts latent representations from the trained VAE.
        
        Returns
        -------
        AnnData
            Updated AnnData object with embeddings stored in `obsm['latent']`.
        """
        self.TZ = []
        self.vae.model.eval()
        with torch.no_grad():
            for x, y in self.vae.total_loader:
                # if on GPU put mini-batch into CUDA memory
                pyro.clear_param_store()
                x = x.to(self.device, non_blocking=True)
                z = self.vae.model.get_latent_representation(x)
                zz = z.cpu().detach().numpy().tolist()
                self.TZ += zz

            self.adata.obsm['latent'] = np.array(self.TZ).astype(np.float32)
        return self.adata
        
    def plotUMAP(self, resolution: float, figure_size: tuple = (8, 6), fontsize: int = 12, cell_size: int = 20) -> None:
        """
        Performs UMAP dimensionality reduction and Leiden clustering on the latent space.

        Parameters
        ----------
        resolution : float
            The resolution parameter for Leiden clustering.
        figure_size : tuple, optional
            Size of the generated UMAP plot (default=(8, 6)).
        fontsize : int, optional
            Font size for labels and legends (default=12).
        cell_size : int, optional
            Size of points in the scatter plot (default=20).

        Returns
        -------
        None
        """
        print("################### Calculating Neighbors #################")
        sc.pp.neighbors(self.adata, n_neighbors=30, use_rep="latent", random_state=self.seed)

        print("################### Calculating UMAP coordinated #################")
        sc.tl.umap(self.adata, random_state=self.seed)

        print("################### Leiden Clustering #################")
        sc.tl.leiden(self.adata, resolution=resolution, random_state=self.seed)

        print("################## Annotating Anndata #################")
        h = self.adata[self.adata.obs.SIDISH == "h"].shape[0]
        b = self.adata[self.adata.obs.SIDISH == "b"].shape[0]
        self.adata.obs["SIDISH_"] = ["High-Risk Cells ({})".format(h) if i == "h" else "Background Cells ({})".format(b) for i in self.adata.obs.SIDISH]
        self.adata.uns["SIDISH__colors"] = np.array(["grey", "red"], dtype="object")

        self.adata.obs["SIDISH__"] = ["High-Risk" if i == "h" else "Background" for i in self.adata.obs.SIDISH]
        self.adata.uns["SIDISH___colors"] = np.array(["grey", "red"], dtype="object")

        print("################## Plotting SIDISH identified High-Risk cells #################")
        plt.figure(figsize=figure_size)
        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["font.size"] = fontsize
        plt.rcParams["legend.fontsize"] = fontsize

        ax = sc.pl.umap(self.adata, color=["SIDISH_"], title="", show=False, size=cell_size,edgecolor="none")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Adjust legend properties: location at the top with more space
        ax.legend(loc="upper center", frameon=False, markerscale=1.5, bbox_to_anchor=(0.5, 1.15))
        plt.tight_layout()
        plt.show()

    def annotateCells(self, test_adata, percentile_cells, mode, perturbation=False):

        adata_new = utils.annotateCells(test_adata, self.deepCox_model.model, percentile_cells, self.device, self.percentile,mode =mode,perturbation=perturbation)

        return adata_new

    def reload(self, path, num_workers = 0):
        self.path = path
        self.num_workers = num_workers
        
        if self.use_spatial_graph and self.spatial_graph is not None:
            print("########################################## Using Spatial Graph in VAE ##########################################")
            self.vae = VAE(epochs=self.epochs_1, adata=self.adata, z_dim=self.latent_size, layer_dims=self.layer_dims, lr=self.lr_1, dropout=self.dropout_1, device=self.device, seed=self.seed, gcn_dims=[32, self.latent_size]) 
            self.vae.initialize(self.adata, W=self.W_matrix, batch_size=self.batch_size, num_workers=self.num_workers, spatial_graph=self.spatial_graph, num_neighbors=self.k_neighbors)
            self.vae.model.load_state_dict(torch.load("{}vae_transfer".format(self.path)))
        
        else:
            print("########################################## Using Dense VAE ##########################################")
            self.vae = VAE(self.epochs_1,self.adata,self.latent_size, self.layer_dims, self.lr_1, self.dropout_1,self.device, self.seed)
            self.vae.initialize(self.adata, self.W_matrix, self.batch_size, self.type, self.num_workers)
            self.vae.model.load_state_dict(torch.load("{}vae_transfer".format(self.path)))


        self.encoder = self.vae
        self.W_vector = self.X_train[:, -1]

        self.deepCox_model = DeepCox(self.X_train, self.y_train, self.W_vector, self.hidden, self.encoder, self.device,self.batch_size, self.seed, self.lr_2, self.dropout_2)
        self.deepCox_model.model.load_state_dict(torch.load("{}deepCox".format(self.path)))

        print("✅ Reload complete – VAE and DeepCox restored")

    def get_percentille(self, percentile):
        self.percentile = percentile
        self.percentile_cells = utils.get_threshold(self.adata, self.deepCox_model.model, self.percentile, self.device)
        return self.percentile_cells
    
    def get_embedding(self, n_neighbors=30, resolution=None, celltype=True):
        if celltype and resolution is not None:
            raise ValueError("Resolution should not be provided when celltype=True.")
        
        self.adata = self.getEmbedding_adata()
        sc.pp.neighbors(self.adata, n_neighbors=n_neighbors, use_rep="latent", random_state=self.seed)
        sc.tl.umap(self.adata, random_state=self.seed)

        if not celltype:
            if resolution is None:
                resolution = 0.8  # Default resolution if not provided
            sc.tl.leiden(self.adata, resolution=resolution, random_state=self.seed)
        
        return self.adata

    def set_adata(self):
        h = self.adata[self.adata.obs.SIDISH == "h"].shape[0]
        b = self.adata[self.adata.obs.SIDISH == "b"].shape[0]
        self.adata.obs["SIDISH_"] = ["High-Risk Cells ({})".format(h) if i == "h" else "Background Cells ({})".format(b) for i in self.adata.obs.SIDISH]
        self.adata.uns["SIDISH__colors"] = np.array(["grey", "red"], dtype="object")

        self.adata.obs["SIDISH__"] = ["High-Risk" if i == "h" else "Background" for i in self.adata.obs.SIDISH]
        self.adata.uns["SIDISH___colors"] = np.array(["grey", "red"], dtype="object")

        fn = "{}adata_SIDISH_embedding.h5ad".format(self.path)
        self.adata.write_h5ad(fn, compression="gzip")

        
    
    def plot_HighRisk_UMAP(self, size= 10, resolution=None, celltype=True):
        self.adata = self.get_embedding(resolution=resolution, celltype=celltype)
        self.set_adata()
        
        plt.figure(figsize=(8, 6))
        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["font.size"] = 12
        plt.rcParams["legend.fontsize"] = 12
        ax = sc.pl.umap(self.adata, color=["SIDISH_"], title="", show=False, size=size, edgecolor="none",)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        # Adjust legend properties: location at the top with more space
        ax.legend(loc="upper center", frameon=False, markerscale=1.5, bbox_to_anchor=(0.5, 1.15))
        plt.tight_layout()
        plt.show()

    def plot_CellType_UMAP(self, size = 10, resolution=None, celltype=True):
        self.adata = self.get_embedding(resolution=resolution, celltype=celltype)
        self.set_adata()
        
        plt.figure(figsize=(8,6))
        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["font.size"] = 12
        plt.rcParams["legend.fontsize"] = 12
        ax = sc.pl.umap(self.adata, color=["leiden"], title="", legend_loc="on data", legend_fontsize=12, show=False, size=size, edgecolor="none", palette='Set3')
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        # Adjust legend properties: location at the top with more space
        ax.legend(loc="upper center", frameon=False, markerscale=1.5, bbox_to_anchor=(0.5, 1.15))
        plt.tight_layout()

    def get_MarkerGenes(self,logfc_threshold=1.5, pval_threshold=0.05, method="wilcoxon", group="h"):
        """
        Identifies marker genes for the specified group using different statistical methods.

        Parameters:
            logfc_threshold (float): Log fold change threshold for filtering significant genes.
            pval_threshold (float): P-value threshold for statistical significance.
            method (str): Method for ranking genes ('wilcoxon', 't-test', 'logreg').
            group (str): The group to compare against others (default is 'h').

        Returns:
            upregulated_genes (list): List of upregulated marker genes.
            downregulated_genes (list): List of downregulated marker genes.
        """

        # Validate method
        supported_methods = ["wilcoxon", "t-test", "logreg"]
        if method not in supported_methods:
            raise ValueError(f"Unsupported method. Choose from {supported_methods}.")
        
        # Normalize and log-transform the data
        sc.pp.normalize_total(self.adata, target_sum=1e4)
        sc.pp.log1p(self.adata)

        # Rank genes using the selected method
        sc.tl.rank_genes_groups(self.adata, groupby="SIDISH", groups=[group], method=method, key_added=f"SIDISH_deg")

        # Extract ranked genes for the specified group
        ranked_genes_df = sc.get.rank_genes_groups_df(self.adata, group=group, key=f"SIDISH_deg")

        # Filter for upregulated and downregulated genes
        self.upregulated_genes = ranked_genes_df[(ranked_genes_df["logfoldchanges"] > logfc_threshold) & (ranked_genes_df["pvals"] < pval_threshold)]["names"].values

        self.downregulated_genes = ranked_genes_df[(ranked_genes_df["logfoldchanges"] < -logfc_threshold) & (ranked_genes_df["pvals"] < pval_threshold)]["names"].values

        return self.upregulated_genes, self.downregulated_genes

    def analyze_perturbation_effects(self):
        self.percent_change, self.p_flip, self.p_score = {}, {}, {}
        self.delta_change = {}
        self.b_dict = {}
        self.b_to_h_dict = {}
        self.h_to_b_dict = {}
        self.h_dict = {}
        
        n_hcells = (self.adata.obs.SIDISH == "h").sum()

        for gene, data in tqdm(zip(self.genes, self.optimized_results), total=len(self.genes), desc="Stats"):

            adata_p = self.annotateCells(data, self.percentile_cells, mode="no", perturbation=True)

            h_to_b = ((self.adata.obs.SIDISH == "h") & (adata_p.obs.SIDISH == "b")).sum()
            b_to_h = ((self.adata.obs.SIDISH == "b") & (adata_p.obs.SIDISH == "h")).sum()

            # % drop in high-risk cells
            self.percent_change[gene] = ((h_to_b - b_to_h) / n_hcells) * 100
            

            # --- 1  one-sided sign test on label flips ---
            flips = h_to_b + b_to_h
            if flips:   # avoid 0-trials edge case
                p_flip = binomtest(k=h_to_b, n=flips, p=0.5, alternative="greater").pvalue
            else:
                p_flip = 1.0
            self.p_flip[gene] = p_flip

            # --- 2  one-sided Wilcoxon on risk-score delta ---
            delta = adata_p.obs["perturbation_score"].values
            self.delta_change[gene] = delta.mean()
            _, p_score = wilcoxon(delta, alternative="greater")
            
            self.h_to_b_dict[gene] = h_to_b
            self.b_to_h_dict[gene] = b_to_h
            self.b_dict[gene] = ((self.adata.obs["SIDISH"] == "b") & (adata_p.obs["SIDISH"] == "b")).sum()
            self.h_dict[gene] = ((self.adata.obs["SIDISH"] == "h") & (adata_p.obs["SIDISH"] == "h")).sum()

            self.p_score[gene] = p_score

        return self.percent_change, self.delta_change, self.p_flip, self.p_score
    
  

    def run_Perturbation(self, n_jobs: int = 4) -> tuple:

    
        self.adata = sc.read_h5ad("{}adata_SIDISH.h5ad".format(self.path))
        self.genes = list(self.adata.var.index)
        
        self.percentage_dict = {}
        self.pvalue_dict = {}
                
        perturbation = InSilicoPerturbation(self.adata)
        perturbation.setup_ppi_network(threshold=0.7)
        self.optimized_results = perturbation.run_parallel_processing(self.adata, n_jobs=4)
        self.percent_change, self.delta_change, self.p_flip, self.p_score = self.analyze_perturbation_effects()
        
        return self.percent_change, self.delta_change, self.p_flip, self.p_score

    def plot_KM(self, penalizer=0.1, data_name="DATA", high_risk_label="High-Risk", background_label="Background", colors=("pink", "grey"), fontsize=12):
        """
        Plot Kaplan-Meier survival curves for High-Risk and background patient groups.

        Parameters:
            penalizer (float): Penalizer for CoxPHFitter regularization.
            data_name (str): Title label for the dataset.
            high_risk_label (str): Label for the High-Risk group.
            background_label (str): Label for the background group.
            colors (tuple): Colors for the survival plots (High-Risk, background).
            fontsize (int): Font size for plot labels and legends.
        """

        # Prepare Data
        DEG_genes = np.append(self.upregulated_genes, ["duration", "event"])
        result = self.bulk.filter(DEG_genes)

        # Fit Cox Proportional Hazards Model
        cph = CoxPHFitter(penalizer=penalizer)
        cph.fit(result, duration_col="duration", event_col="event")

        # Risk Score Calculation
        coef = cph.summary.T.filter(self.upregulated_genes).iloc[0].values.reshape(-1, 1)
        expression = result.iloc[:, :-2].values
        risk_scores = np.dot(expression, coef)

        # Classify Patients into High/Low Risk
        median_risk = np.median(risk_scores)
        risk_group = np.where(risk_scores >= median_risk, high_risk_label, background_label)

        result_df = result.copy()
        result_df["risk"] = risk_group
        result_df["scores"] = risk_scores

        # Log-Rank Test
        low_risk_group = result_df[result_df["risk"] == background_label]
        high_risk_group = result_df[result_df["risk"] == high_risk_label]

        logrank_result = logrank_test(
            durations_A=low_risk_group["duration"],
            durations_B=high_risk_group["duration"],
            event_observed_A=low_risk_group["event"],
            event_observed_B=high_risk_group["event"]
        )
        p_value = logrank_result.p_value

        # Kaplan-Meier Plot
        fig, ax = plt.subplots(figsize=(8, 6))
        kmf = KaplanMeierFitter()

        # Plot High-Risk Group
        kmf.fit(high_risk_group["duration"], event_observed=high_risk_group["event"], label=high_risk_label)
        kmf.plot_survival_function(ax=ax, color=colors[0], ci_show=False, linewidth=2.5)

        # Plot Background Group
        kmf.fit(low_risk_group["duration"], event_observed=low_risk_group["event"], label=background_label)
        kmf.plot_survival_function(ax=ax, color=colors[1], ci_show=False, linewidth=2.5)

        # Aesthetics
        ax.set_title("", fontsize=fontsize)
        ax.set_xlabel("Time (Days)", fontsize=fontsize)
        ax.set_ylabel("Survival Probability", fontsize=fontsize)
        ax.tick_params(axis='both', labelsize=fontsize)
        ax.set_ylim(0, 1)

        # Custom Legend
        legend_labels = [data_name, high_risk_label, background_label]
        legend_handles = [
            plt.Line2D([0], [0], color="w", label=legend_labels[0])
        ] + [
            plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=color, markersize=8, label=label)
            for color, label in zip(colors, legend_labels[1:])
        ]
        fig.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, 1.1), ncol=3, frameon=False, fontsize=fontsize)

        # P-value Formatting
        p_value_formatted = f"P = {p_value:.2e}"
        plt.text(ax.get_xlim()[1] * 0.5, 1.02, p_value_formatted, fontsize=fontsize, ha='center', fontstyle='italic', weight='bold')

        # Final Adjustments
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        plt.show()

    def plot_top_perturbed_genes(self, gene_data, top_n=20):
        """
        Plots a barplot of the top N genes with the highest percentage reduction
        in High-Risk cells after in-silico perturbation.

        Parameters:
        - gene_data (dict): Dictionary of gene perturbation effects.
        - top_n (int): Number of top genes to display. Default is 20.
        """
        # Sort the genes by their reduction percentages
        self.top_genes = dict(sorted(gene_data.items(), key=lambda x: x[1], reverse=True)[:top_n])

        # Plotting
        plt.figure(figsize=(8, 8))
        plt.barh(list(self.top_genes.keys()), list(self.top_genes.values()), color='brown')
        plt.xlabel('% Reduction of High-Risk Cells')
        plt.title(f'Top {top_n} Genes by Reduction in High-Risk Cells After Perturbation')
        plt.gca().invert_yaxis()  # Invert y-axis to show the highest reduction on top
        plt.show()


    def plot_perturbation_UMAP_default(self, genes_of_interest, resolution=None, celltype=True, threshold=0.8):
        """
        Generates UMAP visualizations for specified genes after in-silico perturbation.

        Parameters:
        - adata: AnnData object with latent embeddings.
        - sidish: SIDISH object for annotation and processing.
        - ppi_df: DataFrame containing the PPI network data.
        - genes_of_interest (list): List of genes to visualize.
        - output_path: Filepath for saving the generated UMAP plot.
        - seed: Random seed for reproducibility. Default is 42.
        """
        if not isinstance(genes_of_interest, list):
            raise TypeError("genes_of_interest must be a list of gene names.")

        # Dynamic subplot layout
        n_genes = len(genes_of_interest)
        n_cols = 2
        n_rows = math.ceil(n_genes / n_cols)

        fig, axs = plt.subplots(n_rows, n_cols, figsize=(8 * n_cols, 4.5 * n_rows), squeeze=False)
        axs = axs.flatten()

        for gene, ax in zip(genes_of_interest, axs):
            self.adata = sc.read_h5ad("{}adata_SIDISH.h5ad".format(self.path))
            
            self.ppi_handler = PPINetworkHandler(self.adata)
            self.ppi_handler.load_network(threshold)
            
            direct_neighbors, indirect_neighbors = self.ppi_handler.get_neighbors(gene)
            neighbors = direct_neighbors + indirect_neighbors
            network_df = self.ppi_handler.ppi_df[self.ppi_handler.ppi_df["Source"].isin(neighbors) | self.ppi_handler.ppi_df["Target"].isin(neighbors)]
            
            if not network_df.empty:
                adata_p = GenePerturbationUtils.adjust_expression(self.adata, gene, network_df)
            else:
                adata_p.X = GenePerturbationUtils.knockout_gene(self.adata, gene).tocsr()

            adata_p = self.annotateCells(adata_p, self.percentile_cells, "no")
            h_to_b = ((self.adata.obs['SIDISH'] == 'h') & (adata_p.obs['SIDISH'] == 'b')).sum()
            b_to_h = ((self.adata.obs['SIDISH'] == 'b') & (adata_p.obs['SIDISH'] == 'h')).sum()


            # Calculate the percentage change and store in the dictionary
            percentage_change = ((h_to_b - b_to_h) / (self.adata.obs.SIDISH == "h").values.sum()) * 100
            

            h = adata_p[adata_p.obs.SIDISH == "h"].shape[0]
            b = adata_p[adata_p.obs.SIDISH == "b"].shape[0]

            adata_p.obs["SIDISH_"] = ["High risk cells ({})".format(h) if i == "h" else "Background cells ({})".format(b) for i in adata_p.obs.SIDISH]
            
            self.adata = self.get_embedding(resolution=resolution, celltype=celltype)
            self.set_adata()
            
            umap_combined = pd.DataFrame(self.adata.obsm["X_umap"], columns=["UMAP1", "UMAP2"])
            umap_combined.index = self.adata.obs.index.values
            umap_combined['risk'] = self.adata.obs['SIDISH'].values

            umap_combined['status'] = '{}'.format(((self.adata.obs.SIDISH == "b") & (adata_p.obs.SIDISH == "b")).sum())

            umap_combined.loc[(umap_combined['risk'] == 'h') & (adata_p.obs['SIDISH'] == 'h'), 'status'] = '{}'.format(((self.adata.obs.SIDISH == "h") & (adata_p.obs.SIDISH == "h")).sum())
            umap_combined.loc[(umap_combined['risk'] == 'h') & (adata_p.obs['SIDISH'] == 'b'), 'status'] = '{}'.format(((self.adata.obs.SIDISH == "h") & (adata_p.obs.SIDISH == "b")).sum())
            umap_combined.loc[(umap_combined['risk'] == 'b') & (adata_p.obs['SIDISH'] == 'h'), 'status'] = '{}'.format(((self.adata.obs.SIDISH == "b") & (adata_p.obs.SIDISH == "h")).sum())

            palette = {
                '{}'.format(((self.adata.obs.SIDISH == "b") & (adata_p.obs.SIDISH == "b")).sum()): 'darkgray',
                '{}'.format(((self.adata.obs.SIDISH == "h") & (adata_p.obs.SIDISH == "h")).sum()): 'red',
                '{}'.format(((self.adata.obs.SIDISH == "h") & (adata_p.obs.SIDISH == "b")).sum()): '#3DB1EA',
                '{}'.format(((self.adata.obs.SIDISH == "b") & (adata_p.obs.SIDISH == "h")).sum()): 'purple',
                '{}'.format(percentage_change): 'white'
            }
            

            plot_umap(ax, umap_combined, palette, percentage_change)
            ax.set_title("In-Silico Knockout of {}".format(gene), fontsize=12, y=0.96)
        

        # Remove empty subplots
        if len(axs) > n_genes:
            for ax in axs[n_genes:]:
                ax.axis('off')

        plt.show()
        


    def plot_perturbation_UMAP_differential(self, genes_of_interest, resolution=None, celltype=True, threshold=0.8):
        """
        Generates UMAP visualizations for specified genes after in-silico perturbation.

        Parameters:
        - adata: AnnData object with latent embeddings.
        - sidish: SIDISH object for annotation and processing.
        - ppi_df: DataFrame containing the PPI network data.
        - genes_of_interest (list): List of genes to visualize.
        - output_path: Filepath for saving the generated UMAP plot.
        - seed: Random seed for reproducibility. Default is 42.
        """
        if not isinstance(genes_of_interest, list):
            raise TypeError("genes_of_interest must be a list of gene names.")

        # Dynamic subplot layout
        n_genes = len(genes_of_interest)
        n_cols = 2
        n_rows = math.ceil(n_genes / n_cols)

        fig, axs = plt.subplots(n_rows, n_cols, figsize=(8 * n_cols, 4.5 * n_rows), squeeze=False)
        axs = axs.flatten()

        for gene, ax in zip(genes_of_interest, axs):
            self.adata = sc.read_h5ad("{}adata_SIDISH.h5ad".format(self.path))
            
            self.ppi_handler = PPINetworkHandler(self.adata)
            self.ppi_handler.load_network(threshold)
            
            direct_neighbors, indirect_neighbors = self.ppi_handler.get_neighbors(gene)
            neighbors = direct_neighbors + indirect_neighbors
            network_df = self.ppi_handler.ppi_df[self.ppi_handler.ppi_df["Source"].isin(neighbors) | self.ppi_handler.ppi_df["Target"].isin(neighbors)]
            
            if not network_df.empty:
                adata_p = GenePerturbationUtils.adjust_expression(self.adata, gene, network_df)
            else:
                adata_p.X = GenePerturbationUtils.knockout_gene(self.adata, gene).tocsr()

            adata_p = self.annotateCells(adata_p, self.percentile_cells, mode="no", perturbation=True)


            # --- 2  one-sided Wilcoxon on risk-score delta ---
            delta = adata_p.obs["perturbation_score"].values
            self.delta_change[gene] = delta.mean()

            self.adata = self.get_embedding(resolution=resolution, celltype=celltype)
            self.set_adata()
            
            umap_combined = pd.DataFrame(self.adata.obsm["X_umap"], columns=["UMAP1", "UMAP2"])
            umap_combined.index = adata_p.obs.index.values
            umap_combined['risk'] = adata_p.obs['perturbation_score'].values

            plot_umap_differential(ax, umap_combined)
            ax.set_title("In silico Knockout of {} (Change in score)".format(gene), fontsize=12, y=0.96)
        

        # Remove empty subplots
        if len(axs) > n_genes:
            for ax in axs[n_genes:]:
                ax.axis('off')

        plt.show()


    def run_double_Perturbation(self,genes, top_n = 20, threshold=0.8):

        self.percent_change_double, self.p_flip_double = {}, {}

        self.b_dict_double = {}
        self.b_to_h_dict_double = {}
        self.h_to_b_dict_double = {}
        self.h_dict_double = {}

        pvals_flip = list(self.p_flip.values())

        corrected_pvals_flip = multipletests(pvals_flip, alpha=0.05, method='fdr_bh')

        corrected_pvalue_dict_flip = dict(zip(self.p_flip.keys(), corrected_pvals_flip[1]))

        pvalue_df_flip = pd.DataFrame(list(corrected_pvalue_dict_flip.items()), columns=["Gene", "Pvalue"])
 
        pvalue_df_flip.sort_values(by='Pvalue',ascending=True, inplace=True)

        pvalue_df_flip = pvalue_df_flip[pvalue_df_flip.Pvalue < 0.05]
    
        self.top_genes_flip = pvalue_df_flip.Gene.values
    

        self.percentage_df_flip = pd.DataFrame([self.percent_change], index=None).T.reset_index()
        self.percentage_df_flip.columns = ["Genes", "Scores"]
        self.percentage_df_flip.sort_values(by=["Scores"], ascending=False, inplace=True)


        self.adata = sc.read_h5ad("{}adata_SIDISH.h5ad".format(self.path))

        self.top_genes_flip = self.percentage_df_flip.Genes.values

        self.top_genes = self.top_genes_flip
        print(self.top_genes_flip[:top_n] == genes)

        all_combinations = list(itertools.permutations(self.top_genes[:top_n], 2))
        n_hcells = (self.adata.obs.SIDISH == "h").sum()
        self.percentage_double_dict = {}
        self.pvalue_double_dict = {}
        self.ppi_handler = PPINetworkHandler(self.adata)
        self.ppi_handler.load_network(threshold)
        
        for combination in tqdm(all_combinations[:]):
            adata_p = self.adata.copy()
            for g in combination:
                direct_neighbors, indirect_neighbors = self.ppi_handler.get_neighbors(g)
                neighbors = direct_neighbors + indirect_neighbors
                network_df = self.ppi_handler.ppi_df[self.ppi_handler.ppi_df["Source"].isin(neighbors) | self.ppi_handler.ppi_df["Target"].isin(neighbors)]
                    
                if not network_df.empty:
                    adata_p = GenePerturbationUtils.adjust_expression(self.adata, g, network_df)
                else:
                    adata_p.X = GenePerturbationUtils.knockout_gene(self.adata, g).tocsr()
                    
            adata_p = self.annotateCells(adata_p, self.percentile_cells, mode="no", perturbation=True)

            h_to_b = ((self.adata.obs.SIDISH == "h") & (adata_p.obs.SIDISH == "b")).sum()
            b_to_h = ((self.adata.obs.SIDISH == "b") & (adata_p.obs.SIDISH == "h")).sum()
            
            
            
            self.percent_change_double["{}+{}".format(combination[0], combination[1])] = ((h_to_b - b_to_h) / n_hcells) * 100
            
            # --- 1  one-sided sign test on label flips ---
            flips = h_to_b + b_to_h
            if flips:   # avoid 0-trials edge case
                p_flip_ = binomtest(k=h_to_b, n=flips, p=0.5, alternative="greater").pvalue
            else:
                p_flip_ = 1.0
            self.p_flip_double["{}+{}".format(combination[0], combination[1])] = p_flip_


            self.h_to_b_dict_double["{}+{}".format(combination[0], combination[1])] = h_to_b
            self.b_to_h_dict_double["{}+{}".format(combination[0], combination[1])] = b_to_h
            self.b_dict_double["{}+{}".format(combination[0], combination[1])] = ((self.adata.obs["SIDISH"] == "b") & (adata_p.obs["SIDISH"] == "b")).sum()
            self.h_dict_double["{}+{}".format(combination[0], combination[1])] = ((self.adata.obs["SIDISH"] == "h") & (adata_p.obs["SIDISH"] == "h")).sum()


        return self.percent_change_double, self.p_flip_double



    def run_double_Perturbation_score(self, genes, top_n = 20, threshold=0.8):

        self.p_score_double = {}
        self.delta_change_double = {}

        self.b_dict_double = {}
        self.b_to_h_dict_double = {}
        self.h_to_b_dict_double = {}
        self.h_dict_double = {}

        pvals_score = list(self.p_score.values())
        

        corrected_pvals_score = multipletests(pvals_score, alpha=0.05, method='fdr_bh')

        corrected_pvalue_dict_score = dict(zip(self.p_score.keys(), corrected_pvals_score[1]))

        pvalue_df_score = pd.DataFrame(list(corrected_pvalue_dict_score.items()), columns=["Gene", "Pvalue"])
        
        pvalue_df_score.sort_values(by='Pvalue',ascending=True, inplace=True)
        
        pvalue_df_score = pvalue_df_score[pvalue_df_score.Pvalue < 0.05]
        
        self.top_genes_score = pvalue_df_score.Gene.values

        self.percentage_df_score = pd.DataFrame([self.delta_change], index=None).T.reset_index()
        self.percentage_df_score.columns = ["Genes", "Scores"]
        self.percentage_df_score.sort_values(by=["Scores"], ascending=False, inplace=True)
        self.top_genes_score = self.percentage_df_score.Genes.values

        self.adata = sc.read_h5ad("{}adata_SIDISH.h5ad".format(self.path))
        
        self.top_genes = self.top_genes_score
        print(self.top_genes_score[:top_n] == genes)


        all_combinations = list(itertools.permutations(self.top_genes[:top_n], 2))
        self.percentage_double_dict = {}
        self.pvalue_double_dict = {}
        self.ppi_handler = PPINetworkHandler(self.adata)
        self.ppi_handler.load_network(threshold)
        
        for combination in tqdm(all_combinations[:]):
            adata_p = self.adata.copy()
            for g in combination:
                direct_neighbors, indirect_neighbors = self.ppi_handler.get_neighbors(g)
                neighbors = direct_neighbors + indirect_neighbors
                network_df = self.ppi_handler.ppi_df[self.ppi_handler.ppi_df["Source"].isin(neighbors) | self.ppi_handler.ppi_df["Target"].isin(neighbors)]
                    
                if not network_df.empty:
                    adata_p = GenePerturbationUtils.adjust_expression(self.adata, g, network_df)
                else:
                    adata_p.X = GenePerturbationUtils.knockout_gene(self.adata, g).tocsr()
                    
            adata_p = self.annotateCells(adata_p, self.percentile_cells, mode="no", perturbation=True)
            
            
            # --- 2  one-sided Wilcoxon on risk-score delta ---
            delta_double = adata_p.obs["perturbation_score"].values
            self.delta_change_double["{}+{}".format(combination[0], combination[1])] = delta_double.mean()
            _, p_score_ = wilcoxon(delta_double, alternative="greater")
            self.p_score_double["{}+{}".format(combination[0], combination[1])] = p_score_

        return self.delta_change_double, self.p_score_double

     
    def plot_double_Perturbation_Heatmap(self, percentage_double_dict, top_n=20):
        
        percent_change_df = pd.DataFrame(list(self.percent_change.items()), columns=["Genes", "Scores"])
        percent_change_df.sort_values(by='Scores', ascending=False, inplace=True)
        percent_change_df.Genes.values[:top_n]
            
        df = percent_change_df.iloc[:top_n].sort_values(by='Genes')
        df = df.sort_values(by="Scores", ascending=False)
        
       
        # Filter out zero values and create a DataFrame for the gene pairs
        double_dict = percentage_double_dict
        double_dict = {k: v for k, v in double_dict.items() if v != 0}
        heatmap_data = pd.DataFrame([{'Gene1': k.split('+')[0], 'Gene2': k.split('+')[1], 'Value': v} for k, v in double_dict.items()])
        heatmap_data = heatmap_data.sort_values(by="Value", ascending=True)

        # Pivot the DataFrame to create a matrix suitable for a heatmap
        heatmap_matrix = heatmap_data.pivot(index='Gene1', columns='Gene2', values='Value')
        sorted_genes = df.Genes.values

        # Reordering rows and columns of the matrix
        heatmap_matrix = heatmap_matrix.loc[sorted_genes, sorted_genes]
        
        for gene in sorted_genes:
            score = self.percent_change[gene][id]
            heatmap_matrix[gene] = heatmap_matrix[gene].fillna(score)
        
        # Create the figure and subplots with different width ratios
        fig, (ax1, ax2) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [1, 15]}, figsize=(10, 8))  # Reduce width for 1D heatmap

        # Plot 1D heatmap on the first axis (ax1)
        sns.heatmap(df[['Scores']], cmap='Reds', cbar=True, yticklabels=df['Genes'], xticklabels=False, ax=ax1)
        ax1.set_title('', fontsize=12)
        ax1.set_xlabel('')  # No x-axis label
        ax1.set_ylabel('')

        # Plot 2D heatmap on the second axis (ax2)
        sns.heatmap(heatmap_matrix, cmap='Reds', annot=False, ax=ax2)
        ax2.set_title('')
        ax2.set_xlabel('')
        ax2.set_ylabel('')

        plt.show()



    