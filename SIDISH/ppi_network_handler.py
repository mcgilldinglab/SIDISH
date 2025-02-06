import pandas as pd
import numpy as np
import os

class PPINetworkHandler:
    """Handles PPI network construction and neighbor retrieval using fixed file paths."""

    def __init__(self, adata):
        """
        Initialize the PPI network handler.

        Parameters:
            adata: AnnData
                An AnnData object containing gene expression data.
        """
        self.adata = adata
        self.ppi_df = None

    def load_network(self, threshold=0.7):
        """
        Load and process the PPI network from fixed files (integrating interactions from Hippie and STRING files).

        Fixed files used:
            - Hippie file: located at SIDISH/PPI/hippie_current.txt
            - STRING links file: located at SIDISH/PPI/9606.protein.links.v11.5.txt
            - STRING info file: located at SIDISH/PPI/9606.protein.info.v11.5.txt

        The method performs the following steps:
          1. Builds a gene mapping from STRING info (only including genes present in the AnnData object).
          2. Processes the Hippie file to extract interactions if the score is >= threshold.
          3. Processes the STRING links file to extract interactions if the score is >= threshold * 1000.
          4. Merges the interactions from both sources into one DataFrame.
          5. Constructs a merged network dictionary (with normalized scores) and saves it as a NumPy file.
          6. Returns the merged interactions as a pandas DataFrame.

        Parameters:
            threshold: float, optional (default=0.7)
                Threshold for filtering interactions.

        Returns:
            pd.DataFrame:
                A DataFrame containing the merged PPI network with columns:
                "Source", "Target", and "Weight".
        """
        # Compute paths relative to this file's directory:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        ppi_dir = os.path.join(base_dir, "PPI")
        info_file = os.path.join(ppi_dir, "9606.protein.info.v11.5.txt")
        links_file = os.path.join(ppi_dir, "9606.protein.links.v11.5.txt")
        hippie_file = os.path.join(ppi_dir, "hippie_current.txt")
        
        # Get the set of genes from the AnnData object.
        adatagene = set(self.adata.var.index.values)

        # --- Build gene mapping from STRING info file ---
        with open(info_file, "r") as f:
            lines = f.readlines()[1:]  # Skip header
            gene_map = {
                line.split("\t")[0]: line.split("\t")[1].strip()
                for line in lines
                if line.split("\t")[1].strip() in adatagene
            }

        # --- Process Hippie file ---
        newhippie = []
        with open(hippie_file, "r") as f:
            hippie_lines = f.readlines()
        for line in hippie_lines[1:]:
            parts = line.split("\t")
            if len(parts) < 5:
                continue
            A = parts[0].split("_")[0]
            B = parts[2].split("_")[0]
            try:
                score_value = float(parts[4])
            except ValueError:
                continue
            if A in adatagene and B in adatagene and score_value >= threshold:
                score_int = int(score_value * 1000)  # scale to match STRING file scores
                newhippie.append([A, B, score_int])
                newhippie.append([B, A, score_int])

        # --- Process STRING links file ---
        newstring = []
        with open(links_file, "r") as f:
            string_lines = f.readlines()[1:]  # Skip header
        for line in string_lines:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            try:
                score = int(parts[2].strip("\n"))
            except ValueError:
                continue
            if score >= threshold * 1000:
                if parts[0] in gene_map and parts[1] in gene_map:
                    gene_source = gene_map[parts[0]]
                    gene_target = gene_map[parts[1]]
                    newstring.append([gene_source, gene_target, score])

        # --- Merge interactions from Hippie and STRING sources ---
        merged_interactions = newstring + newhippie
        df = pd.DataFrame(merged_interactions, columns=["Source", "Target", "Weight"])
        df = df.drop_duplicates()

        # --- Build and save the merged network dictionary ---
        merged_dict = {}
        for _, row in df.iterrows():
            src = row["Source"]
            tgt = row["Target"]
            weight = row["Weight"]
            if src not in merged_dict:
                merged_dict[src] = {}
            merged_dict[src][tgt] = weight / 1000.0  # Normalize weight

        self.ppi_df = df
        return self.ppi_df


    def get_neighbors(self, target_gene):
        """
        Retrieve direct and indirect neighbors of a target gene in the PPI network.

        Parameters:
            target_gene: str
                The gene for which to retrieve neighbors.

        Returns:
            tuple:
                A tuple containing:
                  - list of direct neighbors
                  - list of indirect neighbors
        """
        # Direct neighbors: genes connected directly to the target.
        direct_neighbors = set(
            self.ppi_df.loc[self.ppi_df["Source"] == target_gene, "Target"]
        ).union(
            self.ppi_df.loc[self.ppi_df["Target"] == target_gene, "Source"]
        )

        # Indirect neighbors: neighbors of direct neighbors, excluding direct ones and the target gene.
        indirect_neighbors = {
            neighbor2
            for neighbor in direct_neighbors
            for neighbor2 in self.ppi_df.loc[
                (self.ppi_df["Source"] == neighbor) | (self.ppi_df["Target"] == neighbor),
                ["Source", "Target"]
            ].values.flatten()
            if neighbor2 not in direct_neighbors and neighbor2 != target_gene
        }

        return list(direct_neighbors), list(indirect_neighbors)