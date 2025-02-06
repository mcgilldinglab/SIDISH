import numpy as np
from scipy.sparse import lil_matrix

class GenePerturbationUtils:
    """Utility functions for gene perturbation tasks."""

    @staticmethod
    def knockout_gene(adata, gene):
        """
        Knock out a gene's expression by setting its values to zero in the expression matrix.

        Parameters:
            adata: AnnData
                An AnnData object containing gene expression data.
            gene: str
                Name of the gene to knock out.

        Returns:
            lil_matrix:
                A copy of the expression matrix with the specified gene's expression set to zero.
        """
        gene_index = adata.var.index.get_loc(gene)
        X = lil_matrix(adata.X.copy())
        X[:, gene_index] = 0
        return X

    @staticmethod
    def get_cogenes(adata, network_df, genename):
        """
        Retrieve co-expressed genes (neighbors) from the PPI network for a given gene.

        Parameters:
            adata: AnnData
                An AnnData object containing gene expression data.
            network_df: pd.DataFrame
                A DataFrame containing the PPI network information.
            genename: str
                The gene for which to find co-expressed partners.

        Returns:
            list:
                A list of co-expressed gene names (excluding the input gene).
        """
        co_genes = set(network_df["Source"]).union(network_df["Target"]).intersection(adata.var.index)
        co_genes.discard(genename)
        return list(co_genes)

    @staticmethod
    def adjust_expression(adata, genename, network_df):
        """
        Modify gene expression for perturbation by knocking out the target gene and its co-expressed genes.

        Parameters:
            adata: AnnData
                An AnnData object containing gene expression data.
            genename: str
                The gene to knock out.
            network_df: pd.DataFrame
                A DataFrame containing the PPI network information.

        Returns:
            AnnData:
                A new AnnData object with the expression of the target gene and its co-expressed partners set to zero.
        """
        co_genes = GenePerturbationUtils.get_cogenes(adata, network_df, genename)
        X_ = GenePerturbationUtils.knockout_gene(adata, genename)

        if co_genes:
            gene_indices = [adata.var.index.get_loc(gene) for gene in co_genes]
            X_[:, gene_indices] = 0  # Knock out co-expressed genes

        adata_p = adata.copy()
        adata_p.X = X_.tocsr()
        return adata_p
