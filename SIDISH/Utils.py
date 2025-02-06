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

def extractFeature(adata,type='Normal'):
    feature_acc = []

    C = adata.to_df().values
    risk = np.array(adata.obs.SIDISH_value.values)

    train_X, test_X, train_y, test_y = train_test_split(C, risk, test_size=0.2, stratify=risk, random_state=42)
    ss_train = StandardScaler()
    train_X = ss_train.fit_transform(train_X)
    test_X = ss_train.transform(test_X)

    classes = np.unique(train_y)
    class_weights = compute_class_weight(class_weight="balanced", classes=[0, 1], y=train_y)
    class_weight_dict = dict(zip(classes, class_weights))

    ros = RandomUnderSampler(random_state=42)
    train_X, train_y = ros.fit_resample(train_X, train_y)

    #model = LogisticRegression(penalty="l1", solver="liblinear", max_iter=10000,class_weight=class_weight_dict, n_jobs=-1, random_state=42).fit(train_X, train_y)
    model = RandomForestClassifier(criterion='log_loss',n_jobs=-1, class_weight=class_weight_dict,random_state=42).fit(train_X, train_y)
    print('here classifier done')
    return model, train_X, test_X


def sigmoid(x, a=100, b=0):
    value = 1 / (1 + np.exp(-a * (x - b)))
    return value


class Utils():

    def getWeightVector(patients, adata, model, percentile, device, type="Normal"):

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

        params = weibull_min.fit(cell_hazard)
        params_patients = weibull_min.fit(patient_hazard)

        percentile_cells = weibull_min.ppf(percentile, *params)
        percentile_patients = weibull_min.ppf(percentile, *params_patients)

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
        print(percentile_cells)

        return (torch.FloatTensor(val_p), adata, percentile_cells, percentile_cells.max(), percentile_cells.min())

    def annotateCells(adata, model,percentile_cells, device, percentile, mode, type="Normal"):

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
        print('Here get weight')
        explainer = shap.Explainer(model, train_X, seed=seed, feature_names=adata.to_df().columns)
        shap_values = explainer(test_X, check_additivity=False)
        print('Here is done')
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