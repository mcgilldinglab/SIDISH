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

    def load_network(self, threshold=0.8):
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
            threshold: float, optional (default=0.8)
                Threshold for filtering interactions.

        Returns:
            pd.DataFrame:
                A DataFrame containing the merged PPI network with columns:
                "Source", "Target", and "Weight".
        """
        # Compute paths relative to this file's directory:
        hippiefile = open("../data/PPI/hippie_current.txt")
        hippiefile = hippiefile.readlines()
        stringfile = open("../data/PPI/9606.protein.links.v11.5.txt")
        stringfile = stringfile.readlines()
        stringinfo = open("../data/PPI/9606.protein.info.v11.5.txt")
        stringinfo = stringinfo.readlines()

        adatagene = self.adata.var.index.values.tolist()
        stringinfo_dict = {}
        for each in stringinfo[1:]:
            each = each.split("\t")
            if each[1] in adatagene:
                stringinfo_dict[each[0]] = each[1]
        newhippie = []
        for each in hippiefile[1:]:
            each = each.split("\t")
            A = each[0].split("_")[0]
            B = each[2].split("_")[0]
            if A in adatagene and B in adatagene and float(each[4]) >= 0.8:
                newhippie.append([A, B, int(float(each[4]) * 1000)])
                newhippie.append([B, A, int(float(each[4]) * 1000)])
        string_dict = {}
        human_encode_dict = {}
        newstring = []
        for each in stringfile[1:]:
            each = each.split(" ")
            score = int(each[2].strip("\n"))
            if score >= 800:
                if each[0] in stringinfo_dict.keys() and each[1] in stringinfo_dict.keys():
                    gene_source = stringinfo_dict[each[0]]
                    gene_target = stringinfo_dict[each[1]]
                    newstring.append([gene_source, gene_target, score])
        bidirectionalmerge = newstring
        
        merged_dict = {}
        for each in bidirectionalmerge:
            if each[0] not in merged_dict.keys():
                merged_dict[each[0]] = {}
            merged_dict[each[0]][each[1]] = each[2] / 1000
            
        # Convert the dictionary to a list of rows
        data = []
        for source_gene, targets in merged_dict.items():
            for target_gene, weight in targets.items():
                data.append([source_gene, target_gene, weight])

        # Create a DataFrame from the list
        df = pd.DataFrame(data, columns=["Source", "Target", "Weight"])

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
    
        # Find direct neighbors
        direct_neighbors = set(self.ppi_df[self.ppi_df['Source'] == target_gene]['Target'])
        direct_neighbors.update(self.ppi_df[self.ppi_df['Target'] == target_gene]['Source'])

        # Find indirect neighbors: genes connected to direct neighbors but not directly to target_gene
        indirect_neighbors = set()
        for neighbor in direct_neighbors:
            # Find neighbors of direct neighbors
            second_degree = set(self.ppi_df[self.ppi_df['Source'] == neighbor]['Target'])
            second_degree.update(self.ppi_df[self.ppi_df['Target'] == neighbor]['Source'])
            # Exclude direct neighbors and the target_gene itself
            indirect_neighbors.update(second_degree - direct_neighbors - {target_gene})

        return list(direct_neighbors), list(indirect_neighbors)