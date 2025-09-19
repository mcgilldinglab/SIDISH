import torch
import numpy as np
from scipy.stats import weibull_min
from imblearn.over_sampling import RandomOverSampler
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from imblearn.under_sampling import RandomUnderSampler
import copy
import shap
import pandas as pd
from scipy import stats
from typing import Literal
from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import kneighbors_graph
import torch_geometric.utils as pyg_utils
from torch_geometric.data import Data
from scipy.stats import binomtest

def extractFeature(adata,type='Normal'):
    feature_acc = []

    C = adata.to_df().values
    risk = np.array(adata.obs.SIDISH_value.values)

    train_X, test_X, train_y, test_y = train_test_split(C, risk, test_size=0.2, stratify=risk, random_state=42)
    ss_train = StandardScaler()
    train_X = ss_train.fit_transform(train_X)
    test_X = ss_train.transform(test_X)

    classes = np.unique(train_y)
    class_weights = compute_class_weight(class_weight="balanced", classes=np.array([0, 1]), y=train_y)
    class_weight_dict = dict(zip(classes, class_weights))

    ros = RandomUnderSampler(random_state=42)
    train_X, train_y = ros.fit_resample(train_X, train_y)

    #model = LogisticRegression(penalty="l1", solver="liblinear", max_iter=10000,class_weight=class_weight_dict, n_jobs=-1, random_state=42).fit(train_X, train_y)
    model = RandomForestClassifier(criterion='log_loss',n_jobs=-1, class_weight=class_weight_dict,random_state=42).fit(train_X, train_y)
    return model, train_X, test_X


def sigmoid(x, a=100, b=0):
    value = 1 / (1 + np.exp(-a * (x - b)))
    return value

def create_spatial_graph(spatial_coords, method='knn', k=5, radius=None, include_self=False):
    """
    Create a graph from spatial coordinates with limited neighborhood size
    
    Parameters:
    -----------
    spatial_coords : numpy.ndarray
        Array of shape [n_cells, n_dims] containing spatial coordinates
    method : str
        Method to create the graph, either 'knn' or 'radius'
    k : int
        Number of nearest neighbors for KNN graph (default: 5)
    radius : float
        Radius for radius graph
    include_self : bool
        Whether to include self-loops
    
    Returns:
    --------
    edge_index : torch.LongTensor
        Edge indices in COO format [2, num_edges]
    edge_weight : torch.FloatTensor
        Edge weights based on distance [num_edges]
    """
    n_cells = spatial_coords.shape[0]
    
    if method == 'knn':
        # Create KNN graph with limited neighborhood size
        adjacency = kneighbors_graph(
            spatial_coords, k, mode='distance', include_self=include_self
        )
        
    elif method == 'radius':
        if radius is None:
            raise ValueError("Radius must be provided for radius graph method")
        
        # Calculate pairwise distances
        dist_matrix = squareform(pdist(spatial_coords))
        
        # Create binary adjacency matrix based on radius
        adjacency = (dist_matrix <= radius).astype(np.float32)
        
        # Remove self-loops if needed
        if not include_self:
            np.fill_diagonal(adjacency, 0)
    else:
        raise ValueError(f"Unknown graph creation method: {method}")
    
    # Convert to edge_index and edge_weight
    adjacency_coo = adjacency.tocoo()
    edge_index = torch.LongTensor(np.vstack((adjacency_coo.row, adjacency_coo.col)))
    edge_weight = torch.FloatTensor(adjacency_coo.data)
    
    # Convert weights to similarity (1 / (distance + epsilon))
    edge_weight = 1.0 / (edge_weight + 1e-6)
    
    return edge_index, edge_weight

def get_spatial_graph_from_adata(adata, spatial_key='spatial', method='knn', k=5, radius=None, include_self=False):
    """
    Create a graph from spatial coordinates stored in AnnData object
    
    Parameters:
    -----------
    adata : AnnData
        AnnData object containing spatial coordinates
    spatial_key : str
        Key in adata.obsm where spatial coordinates are stored
    method : str
        Method to create the graph, either 'knn' or 'radius'
    k : int
        Number of nearest neighbors for KNN graph (default: 5)
    radius : float
        Radius for radius graph
    include_self : bool
        Whether to include self-loops
        
    Returns:
    --------
    edge_index : torch.LongTensor
        Edge indices in COO format [2, num_edges]
    edge_weight : torch.FloatTensor
        Edge weights based on distance [num_edges]
    """
    if spatial_key not in adata.obsm:
        raise ValueError(f"Spatial coordinates not found in adata.obsm['{spatial_key}']")
    
    spatial_coords = adata.obsm[spatial_key]
    return create_spatial_graph(spatial_coords, method, k, radius, include_self)

def create_pytorch_geometric_data(adata, spatial_key='spatial', method='knn', k=5):
    """
    Create a PyTorch Geometric Data object from AnnData with spatial information
    
    Parameters:
    -----------
    adata : AnnData
        AnnData object containing gene expression and spatial data
    spatial_key : str
        Key in adata.obsm where spatial coordinates are stored
    method : str
        Method to create the graph, either 'knn' or 'radius'
    k : int
        Number of nearest neighbors for KNN graph (default: 5)
        
    Returns:
    --------
    data : torch_geometric.data.Data
        PyTorch Geometric Data object containing the graph
    """
    # Extract expression data
    if isinstance(adata.X, np.ndarray):
        x = torch.from_numpy(adata.X).float()
    else:
        x = torch.from_numpy(adata.X.todense()).float()
    
    # Create spatial graph
    edge_index, edge_attr = get_spatial_graph_from_adata(
        adata, 
        spatial_key=spatial_key, 
        method=method, 
        k=k
    )
    
    # Create Data object
    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr
    )
    
    return data


def r_squared(sample, dist, params):
    x_sorted  = np.sort(sample)
    n         = len(sample)
    ecdf      = np.arange(1, n + 1) / n
    model_cdf = dist.cdf(x_sorted, *params)
    sse = np.sum((ecdf - model_cdf) ** 2)
    sst = np.sum((ecdf - ecdf.mean()) ** 2)
    return 1 - sse / sst


class Utils():

    def getWeightVector(patients, adata, model, percentile: float, device: str, distribution: Literal["default", "fitted"] = "default", dist = None):

        labels = []
        p = patients
        all_p = p.to(device, non_blocking=True)
        val = model(all_p).detach().cpu().flatten()

        X = adata.to_df().values

        # X = np.log(1+X) #MinMaxScaler().fit_transform(np.log(1 + X))
        all_X = torch.from_numpy(X).type(torch.float32)
        all_X = all_X.to(device, non_blocking=True)
        val_sc = model(all_X).detach().cpu().flatten()

        ##### Cell high risk labeling #####
        cell_hazard = val_sc
        patient_hazard = val
        
        DISTROS = { "Weibull": stats.weibull_min, "Exponential": stats.expon, "Gamma" : stats.gamma }
            
        returned_dist =None
        if distribution == "default":
            params = weibull_min.fit(cell_hazard)
            params_patients = weibull_min.fit(patient_hazard)

            percentile_cells = weibull_min.ppf(percentile, *params)
            percentile_patients = weibull_min.ppf(percentile, *params_patients)
        
        else:
            if dist is not None:
                params = DISTROS[dist].fit(cell_hazard)
                params_patients = DISTROS[dist].fit(patient_hazard)

                percentile_cells = DISTROS[dist].ppf(percentile, *params)
                percentile_patients = DISTROS[dist].ppf(percentile, *params_patients)
                
            else:
                
                exp_cell_hazard = np.exp(cell_hazard)
                
                results = []
                for idx, (label, dist) in enumerate(DISTROS.items()):
                    # 4.1  Fit parameters
                    if label != "Weibull":
                        params = dist.fit(exp_cell_hazard, floc=0)
                    else:
                        params = dist.fit(exp_cell_hazard)
                    
                    # 4.2  Metrics
                    logL = np.sum(dist.logpdf(exp_cell_hazard, *params))
                    k, n = len(params), len(exp_cell_hazard)
                    aic  = 2 * k - 2 * logL
                    bic  = k * np.log(n) - 2 * logL
                    ks_D, ks_p = stats.kstest(exp_cell_hazard, dist.name, args=params)
                    r2 = r_squared(exp_cell_hazard, dist, params)

                    results.append(dict(distribution=label, logL=logL, AIC=aic, BIC=bic, KS_D=ks_D, KS_p=ks_p, R2=r2, params=params))
                df = pd.DataFrame(results).round(6)
                    
                # Hybrid rank-sum (lower is better)
                ranked = df.assign(
                    r_aic = df["AIC"].rank(method="min", ascending=True),
                    r_bic = df["BIC"].rank(method="min", ascending=True),
                    r_ks  = df["KS_D"].rank(method="min", ascending=True),
                    r_r2  = df["R2"].rank(method="min", ascending=False),
                )
                ranked["rank_sum"] = ranked[["r_aic","r_bic","r_ks","r_r2"]].sum(axis=1)

                # Pick best by rank_sum, tie-break by AIC
                best_idx = ranked.sort_values(["rank_sum","AIC"], ascending=[True, True]).index[0]
                best_hybrid = df.loc[best_idx, "distribution"]
                returned_dist = best_hybrid
                print(best_hybrid)
                print("Best Distribution: ".format(best_hybrid))
                    
                params = DISTROS[best_hybrid].fit(cell_hazard)
                params_patients = DISTROS[best_hybrid].fit(patient_hazard)
                    
                percentile_cells = DISTROS[best_hybrid].ppf(percentile, *params)
                percentile_patients = DISTROS[best_hybrid].ppf(percentile, *params_patients)

        high_risk_cells = cell_hazard >= percentile_cells
        high_risk_cells_ = high_risk_cells.type(torch.int)
        mask_patients = patient_hazard >= percentile_patients

        adata.obs["SIDISH_value"] = high_risk_cells_
        adata.obs["risk_value"] = val_sc

        for i in adata.obs.SIDISH_value:
            if i == 1:
                labels.append("h")
            else:
                labels.append("b")

        adata.obs["SIDISH"] = labels

        val_p = torch.sigmoid(val)
        val_p[~mask_patients] = 0.0

        return (torch.FloatTensor(val_p), adata, percentile_cells, percentile_cells.max(), percentile_cells.min(), returned_dist)
    
    def annotateCells(adata, model,percentile_cells, device, percentile, mode, perturbation=False):

        labels = []
        # X = np.log(1 + X)
        X = adata.to_df().values
        all_X = torch.from_numpy(X).type(torch.float32)
        all_X = all_X.to(device, non_blocking=True)
        val_sc = model(all_X).detach().cpu().flatten()

        ##### Cell high risk labeling #####
        cell_hazard = val_sc
        if mode == 'test':
            params = weibull_min.fit(cell_hazard)
            percentile_cells = weibull_min.ppf(percentile, *params)
        else:
            percentile_cells = percentile_cells

        high_risk_cells = cell_hazard >= percentile_cells
        high_risk_cells_ = high_risk_cells.type(torch.int)
        
        if perturbation:
            old_risk_score = adata.obs["risk_value"].values
            
            delta_score = old_risk_score - val_sc.numpy()
            adata.obs['perturbation_score'] = delta_score
            adata.obs['perturbation_score_abs'] = np.abs(delta_score)
            
            adata.obs["SIDISH_value"] = high_risk_cells_
            adata.obs["risk_value"] = val_sc
        
        else: 
            adata.obs["SIDISH_value"] = high_risk_cells_
            adata.obs["risk_value"] = val_sc

        for i in adata.obs.SIDISH_value:
            if i == 1:
                labels.append("h")
            else:
                labels.append("b")

        adata.obs["SIDISH"] = labels
        return  adata

    def getWeightMatrix(adata, seed, steepness=100,type='Normal'):
        model, train_X, test_X = extractFeature(adata, type)
        explainer = shap.Explainer(model, train_X, seed=seed, feature_names=adata.to_df().columns)
        shap_values = explainer(test_X, check_additivity=False)
        shap_values_class_1 = shap_values.values[:, :, 0]
        shap_values = pd.DataFrame(copy.deepcopy(shap_values_class_1), columns=adata.to_df().columns)
        q = shap_values.mean(axis=0)

        W = []
        for i in adata.obs.SIDISH:
            if i in "h":
                x = q.values
                weights_ = sigmoid(x, steepness)
                W.append(weights_)

            else:
                W.append(np.zeros(adata.shape[1]))
        W = np.array(W)

        return W

    def get_threshold(adata, model, percentile, device):

        X = adata.to_df().values
        # X = np.log(1+X) #MinMaxScaler().fit_transform(np.log(1 + X))
        all_X = torch.from_numpy(X).type(torch.float32)
        all_X = all_X.to(device, non_blocking=True)
        val_sc = model(all_X).detach().cpu().flatten()

        ##### Cell high risk labeling #####
        cell_hazard = val_sc

        params = weibull_min.fit(cell_hazard)

        percentile_cells = weibull_min.ppf(percentile, *params)
        print(percentile_cells)

        return percentile_cells