from SIDISH.DEEP_COX import DEEPCOX as DeepCox
from SIDISH.VAE import VAE as VAE
from SIDISH.Utils import Utils as utils
from SIDISH.in_silico_perturbation import InSilicoPerturbation
import pandas as pd
import numpy as np
import torch
import scanpy as sc
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import os
from tqdm import tqdm
from scipy.stats import chi2_contingency
import scanpy as sc
import matplotlib.pyplot as plt
from lifelines import CoxPHFitter
from lifelines.statistics import logrank_test
from lifelines import KaplanMeierFitter

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

class SIDISH:
    """
    SIDISH (Semi-Supervised Iterative Deep Learning for Identifying High-Risk Cells).

    This framework integrates single-cell and bulk RNA-seq data to identify 
    high-risk cancer cells and potential biomarkers.

    Parameters
    ----------
    adata : AnnData
        Single-cell RNA-seq data.
    bulk : pd.DataFrame
        Bulk RNA-seq data.
    device : str
        Computation device ('cpu' or 'cuda').
    seed : int, optional
        Random seed for reproducibility (default=1234).
    """

    def __init__(self, adata, bulk, device: str, seed: int = 1234) -> None:
        self.adata = adata
        self.bulk = bulk
        self.device = device
        self.seed = seed

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
        self.W_matrix_uncapped = np.ones(self.adata.X.shape)

    def init_Phase2(self, epochs: int, hidden: int, lr: float, dropout: float, test_size: float, batch_size: int) -> None:
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
        batch_size : int
            Number of samples per batch.

        Returns
        -------
        None
        """
        self.epochs_2 = epochs
        self.hidden = hidden
        # self.iterations = iterations
        self.lr_2 = lr
        self.dropout_2 = dropout
        self.batch_size = batch_size

        # Initialise the weight  vector of phase 2
        self.W_vector = np.ones(self.bulk.iloc[:,2:].shape[0])
        self.bulk['weight'] = self.W_vector

        self.X = self.bulk.iloc[:, 2:].values
        self.Y = self.bulk.iloc[:, :2].values
        self.X_train, self.X_test, self.y_train, self.y_test = process_Data(self.X, self.Y, test_size, batch_size, self.seed)

    def train(self, iterations: int, percentile: float, steepness: float, path: str, num_workers: int = 8, show: bool = True) -> sc.AnnData:
        """
        Trains the SIDISH framework iteratively, refining the identification of high-risk cells.

        This function iteratively updates high-risk cell classifications by integrating 
        single-cell and bulk RNA-seq data. Each iteration includes:
        - Training the VAE model on single-cell data.
        - Training the Deep Cox model on bulk RNA-seq survival data.
        - Updating weight matrices to improve high-risk cell identification.

        Parameters
        ----------
        iterations : int
            Number of training iterations.
        percentile : float
            Threshold percentile for defining high-risk cells.
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
            Updated AnnData object containing the refined high-risk cell classifications.
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
        self.vae = VAE(self.epochs_1,self.adata,self.latent_size, self.layer_dims, self.optimizer, self.lr_1, self.dropout_1,self.device, self.seed)
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
            self.scores, self.adata_, self.percentile_cells, self.cells_max, self.cells_min = utils.getWeightVector(patients_data, self.vae.adata, self.deepCox_model.model, self.percentile, self.device, self.type)
            self.percentile_list.append(self.percentile_cells)
            self.W_vector += self.scores
            self.W_temp = utils.getWeightMatrix(self.adata_, self.seed, self.steepness, self.type)
            self.W_matrix += self.W_temp
            self.W_matrix[self.W_matrix >= 2] = 2

            self.W_matrix_uncapped += self.W_temp

            self.adata = self.adata_.copy()
            pd.DataFrame(self.W_matrix).to_csv("{}W_matrix_{}.csv".format(self.path,i))
            #pd.DataFrame(self.W_matrix_uncapped).to_csv("{}W_matrix_uncapped_{}.csv".format(path, i))

            if i == (iterations - 1):
                print("########################################## SIDISH TRAINING DONE ##########################################")
                break

            else:
                print("########################################## ITERATION {} OUT OF {} ##########################################".format(i+2, iterations))
                self.vae = VAE(self.epochs_3,self.adata,self.latent_size, self.layer_dims, self.optimizer, self.lr_3, self.dropout_1,self.device, self.seed)
                self.vae.initialize(self.adata, self.W_matrix, self.batch_size, self.type,self.num_workers)
                self.vae.model.load_state_dict(torch.load("{}vae_transfer".format(self.path)))
                self.vae.train()

                # Save VAE for iterative process
                torch.save(self.vae.model.state_dict(), "{}vae_transfer".format(self.path))

        torch.save(self.deepCox_model.model.state_dict(), "{}deepCox".format(self.path))

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

    def annotateCells(self, test_adata, percentile_cells, mode, type ='Normal' ):

        adata_new = utils.annotateCells(test_adata, self.deepCox_model.model, percentile_cells, self.device, self.percentile,mode =mode, type=type)

        return adata_new

    def reload(self, path, num_workers = 8):
        self.path = path
        self.num_workers = num_workers
        self.vae = VAE(self.epochs_3,self.adata,self.latent_size, self.layer_dims, self.optimizer, self.lr_3, self.dropout_1,self.device, self.seed)
        self.vae.initialize(self.adata, self.W_matrix, self.batch_size, self.type, self.num_workers)
        self.vae.model.load_state_dict(torch.load("{}vae_transfer".format(self.path)))

        self.encoder = self.vae

        self.W_vector = self.X_train[:, -1]

        self.deepCox_model = DeepCox(self.X_train, self.y_train, self.W_vector, self.hidden, self.encoder, self.device,self.batch_size, self.seed, self.lr_2, self.dropout_2)
        self.deepCox_model.model.load_state_dict(torch.load("{}deepCox".format(self.path)))

        print('Reload Complete')    

    def get_percentille(self, percentile):
        self.percentile = percentile
        self.percentile_cells = utils.get_threshold(self.adata, self.deepCox_model.model, self.percentile, self.device)
        return self.percentile_cells
    
    def get_embedding(self, n_neighbors=30, resolution=None, celltype=True):
        self.adata = self.getEmbedding()
        if celltype and resolution is not None:
            raise ValueError("Resolution should not be provided when celltype=True.")

        self.adata = self.vae.getEmbedding()
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

        return self.adata
    
    def plot_HighRisk_UMAP(self, size= 10):
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

    def plot_CellType_UMAP(self, size = 10):
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
        """
        Perform in-silico perturbation statistical analysis.

        Returns:
            tuple:
                A tuple containing:
                  - percentage_dict: dict mapping gene to percentage change.
                  - pvalue_dict: dict mapping gene to chi-squared test p-value.
        """
        percentage_dict = {}
        pvalue_dict = {}
        b_dict = {}
        b_to_h_dict = {}
        h_to_b_dict = {}
        h_dict = {}

        self.genes = list(self.adata.var.index)

        n_hcells = (self.adata.obs.SIDISH == "h").sum()

        for gene, data in tqdm(
            zip(self.genes, self.optimized_results),
            total=len(self.genes),
            desc="Calculating Stats"
        ):
            # Annotate the perturbed cells
            adata_p = self.annotateCells(data, self.percentile_cells, "no")

            # Count cell state transitions
            h_to_b = ((self.adata.obs["SIDISH"] == "h") & (adata_p.obs["SIDISH"] == "b")).sum()
            b_to_h = ((self.adata.obs["SIDISH"] == "b") & (adata_p.obs["SIDISH"] == "h")).sum()

            h_to_b_dict[gene] = h_to_b
            b_to_h_dict[gene] = b_to_h
            b_dict[gene] = ((self.adata.obs["SIDISH"] == "b") & (adata_p.obs["SIDISH"] == "b")).sum()
            h_dict[gene] = ((self.adata.obs["SIDISH"] == "h") & (adata_p.obs["SIDISH"] == "h")).sum()

            # Compute percentage change
            percentage_dict[gene] = ((h_to_b - b_to_h) / n_hcells) * 100

            # Construct contingency table and perform chi-squared test
            val = n_hcells - (h_to_b - b_to_h)
            contingency_table = [
                [n_hcells, self.adata.shape[0] - n_hcells],
                [val, adata_p.shape[0] - val]
            ]
            _, p_value, _, _ = chi2_contingency(contingency_table, correction=True)
            pvalue_dict[gene] = p_value

        return percentage_dict, pvalue_dict


    def run_Perturbation(self, n_jobs: int = 4) -> tuple:
        """
        Runs gene perturbation simulations using the in-silico perturbation module.

        Parameters
        ----------
        n_jobs : int, optional
            Number of parallel jobs for perturbation simulations (default=4).

        Returns
        -------
        tuple
            - dict: Mapping of genes to percentage change in high-risk cells.
            - dict: Mapping of genes to chi-squared test p-values.
        """
        perturbation = InSilicoPerturbation(self.adata)
        perturbation.setup_ppi_network(threshold=0.7)
        self.optimized_results = perturbation.run_parallel_processing(n_jobs=4)
        percentage_dict, pvalue_dict = self.analyze_perturbation_effects()
        return percentage_dict, pvalue_dict

    def plot_KM(self, penalizer=0.1, data_name="DATA", high_risk_label="High-Risk", background_label="Background", colors=("pink", "grey"), fontsize=12):
        """
        Plot Kaplan-Meier survival curves for high-risk and background patient groups.

        Parameters:
            penalizer (float): Penalizer for CoxPHFitter regularization.
            data_name (str): Title label for the dataset.
            high_risk_label (str): Label for the high-risk group.
            background_label (str): Label for the background group.
            colors (tuple): Colors for the survival plots (high-risk, background).
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

