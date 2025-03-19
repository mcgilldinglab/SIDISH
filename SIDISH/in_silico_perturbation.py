import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
from scipy.stats import chi2_contingency

from SIDISH.gene_perturbation_utils import GenePerturbationUtils
from SIDISH.ppi_network_handler import PPINetworkHandler

class InSilicoPerturbation:
    """
    Handles single-cell in-silico perturbation experiments.

    Attributes:
        adata: AnnData
            The original gene expression data.
        sidish: Object
            An object providing cell annotation functionality.
        genes: list
            A list of gene names from the AnnData object.
        ppi_handler: PPINetworkHandler
            An instance of PPINetworkHandler for managing the PPI network.
    """

    def __init__(self, adata):
        """
        Initialize the in-silico perturbation experiment.

        Parameters:
            adata: AnnData
                An AnnData object containing gene expression data.
        """
        self.adata = adata
        self.genes = list(adata.var.index)
        self.ppi_handler = PPINetworkHandler(adata)
        self.optimized_results = None

    def setup_ppi_network(self, threshold=0.8):
        """
        Initialize the PPI network.

        Parameters:
            hippie_path: str
                Path to the HIPPIE file.
            string_path: str
                Path to the STRING file.
            info_path: str
                Path to the gene mapping info file.
            threshold: float, optional (default=0.7)
                Threshold value for filtering interactions.

        Returns:
            pd.DataFrame:
                The loaded and processed PPI network.
        """
        return self.ppi_handler.load_network(threshold)

    def process_gene(self, adata, gene):
        """
        Process a single gene for perturbation by knocking it out along with its network neighbors.

        Parameters:
            gene: str
                The gene to knock out.

        Returns:
            AnnData:
                A new AnnData object representing the perturbed state with the gene (and its neighbors) knocked out.
        """
        direct_neighbors, indirect_neighbors = self.ppi_handler.get_neighbors(gene)
        neighbors = direct_neighbors + indirect_neighbors

        network_df = self.ppi_handler.ppi_df[
            self.ppi_handler.ppi_df["Source"].isin(neighbors) | 
            self.ppi_handler.ppi_df["Target"].isin(neighbors)
        ]

        if network_df.empty:
            adata_p = adata.copy()
            X_ = GenePerturbationUtils.knockout_gene(adata, gene)
            adata_p.X = X_.tocsr()
        else:
            adata_p = GenePerturbationUtils.adjust_expression(adata, gene, network_df)

        return adata_p

    def run_parallel_processing(self, adata, genes, n_jobs=4):
        """
        Run gene perturbation processing in parallel.

        Parameters:
            n_jobs: int, optional (default=4)
                Number of parallel jobs.

        Side Effects:
            Sets the 'optimized_results' attribute with the list of perturbed AnnData objects.
        """
        self.optimized_results = Parallel(n_jobs=n_jobs)(
            delayed(self.process_gene)(adata, gene)
            for gene in tqdm(genes, desc="Processing Genes", leave=True)
        )

        return self.optimized_results
