from DEEP_COX import DEEPCOX as DeepCox
from VAE import VAE as VAE
from Utils import Utils as utils
import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import os

def process_Data(X, Y, test_size, batch_size, seed):

    """
    Processes the Bulk data by splitting it into training and testing datasets, converting them to tensors, and creating data loaders for training and testing.

    Parameters
    ----------
    X : array-like
        Bulk gene expression data.
    Y : array-like
        Survival + weight data, i.e [survival days, event, weight]
    test_size : float
        Proportion of the dataset to include in the test split.
    batch_size : int
        Number of patients per batch to load.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    train_loader : DataLoader
        DataLoader for the training dataset.
    test_loader : DataLoader
        DataLoader for the testing dataset.
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

    def __init__(self, adata, bulk,  device, seed=1234):
        self.adata = adata
        self.bulk = bulk
        self.device = device
        self.seed = seed

    def init_Phase1(self, epochs, i_epochs, latent_size, layer_dims, batch_size, optimizer, lr, lr_3, dropout, type = 'Normal'):
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

    def init_Phase2(self, epochs, hidden, lr, dropout, test_size, batch_size):
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

    def train(self, iterations, percentile, steepness, path, num_workers = 8):
        os.makedirs(path, exist_ok=True)
        
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
        torch.save(self.vae.model.state_dict(), "{}vae_transfer".format(path))

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
            pd.DataFrame(self.W_matrix).to_csv("{}W_matrix_{}.csv".format(path,i))
            #pd.DataFrame(self.W_matrix_uncapped).to_csv("{}W_matrix_uncapped_{}.csv".format(path, i))

            if i == (iterations - 1):
                break

            else:
                print("########################################## ITERATION {} OUT OF {} ##########################################".format(i, iterations))
                self.vae = VAE(self.epochs_3,self.adata,self.latent_size, self.layer_dims, self.optimizer, self.lr_3, self.dropout_1,self.device, self.seed)
                self.vae.initialize(self.adata, self.W_matrix, self.batch_size, self.type,self.num_workers)
                self.vae.model.load_state_dict(torch.load("{}vae_transfer".format(path)))
                self.vae.train()

                # Save VAE for iterative process
                torch.save(self.vae.model.state_dict(), "{}vae_transfer".format(path))

        torch.save(self.deepCox_model.model.state_dict(), "{}deepCox".format(path))

        fn = "{}adata_SIDISH.h5ad".format(path)
        self.adata.write_h5ad(fn, compression="gzip")
        return self.adata

    def annotateCells(self, test_adata, percentile_cells, mode, type ='Normal' ):

        adata_new = utils.annotateCells(test_adata, self.deepCox_model.model, percentile_cells, self.device, self.percentile,mode =mode, type=type)

        return adata_new

    def reload(self, path):

        self.vae = VAE(self.epochs_3,self.adata,self.latent_size, self.layer_dims, self.optimizer, self.lr_3, self.dropout_1,self.device, self.seed)
        self.vae.initialize(self.adata, self.W_matrix, self.batch_size, self.type, self.num_workers)
        self.vae.model.load_state_dict(torch.load("{}vae_transfer".format(path)))

        self.encoder = self.vae

        self.W_vector = self.X_train[:, -1]

        self.deepCox_model = DeepCox(self.X_train, self.y_train, self.W_vector, self.hidden, self.encoder, self.device,self.batch_size, self.seed, self.lr_2, self.dropout_2)
        self.deepCox_model.model.load_state_dict(torch.load("{}deepCox".format(path)))

        print('Reload Complete')    

    def get_percentille(self, percentile):
        self.percentile = percentile
        self.percentile_cells = utils.get_threshold(self.adata, self.deepCox_model.model, self.percentile, self.device)
        return self.percentile_cells