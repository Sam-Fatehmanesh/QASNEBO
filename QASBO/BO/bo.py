import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
from torch.utils.data import DataLoader, random_split, TensorDataset
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from QASBO.models.neural_predictors import Neural_Predictor
from torch import Tensor
from torch.distributions.normal import Normal

class BayesOpt:
    def surrogate_model_train(self, x_train: Tensor, y_train: Tensor, x_pred: Tensor):
        
        pass

    def surrogate_model_inference(self, x: Tensor):
        
        raise NotImplementedError

    def objective(self, x: Tensor):
        
        raise NotImplementedError
    
    def acquisition_function(self, x: Tensor):
        
        raise NotImplementedError

    def sample_search_space(self):

        raise NotImplementedError

    def bo_run(self):

        raise NotImplementedError

# Neural Evolutionary Bayesian Optimization
class nebo(BayesOpt):
    def surrogate_model_train(self, 
    x_train, 
    y_train, 
    dropout_p = 0.25, 
    dataloader_thread_count = 3, 
    training_epochs = 2, 
    np_trainer_device = 'gpu', 
    verbose = False
    ):
        # resets model weights for training
        self.model.reset_weights()

        # init dataloader 
        train_dl = DataLoader(dataset = TensorDataset(x_train, y_train), 
        shuffle = True, 
        batch_size = 1, 
        num_workers = dataloader_thread_count)

        # init trainer
        trainer = pl.Trainer(max_epochs = training_epochs, 
        accelerator = np_trainer_device, 
        enable_model_summary = verbose, 
        enable_progress_bar = verbose)

        # training of the neural network predictor
        trainer.fit(self.model, train_dataloaders = train_dl)
    
    def surrogate_model_inference(self, x_pred, mciterations):
        # returns model inference output in the form of mean, variances
        mean, variances = self.model.forward_mcd(x_pred, mciterations = mciterations)

        return mean, variances

    def acquisition_function(self, means, variances, y_max, best_count = 1):
        # expected improvement constant
        ei_constant = 0.5

        # converts var to std and multiply by a calibration cosntant
        stds = torch.sqrt(variances) * ei_constant

        # normal distribution
        dist = Normal(means, stds)
        # Z score
        Ymax = y_max
        Z = (Ymax - means) * stds.reciprocal()
        # cumulative distributions
        cd = dist.cdf(torch.squeeze(Ymax))
        # probability densities
        pd = dist.log_prob(Ymax).exp()
        # computes expected improvements
        ei = (-stds) * (Z * cd + pd)
        # sorted ei indicies
        ei_sorted, ei_sorted_indices = torch.topk(input = ei, dim = 0, k = best_count)

        # returns expected improvements and their corresponding sorted indicies
        return ei_sorted_indices

    def bo_run(self, iterations, search_space_sample_size, objective_batch_size = 1):
        # collects a sample from the search space
        self.X_max_list = self.sample_search_space(quantity = 1)
        # computes the objective value of a single sample to serve as the first maximum
        self.Y_max_list = self.objective(self.X_max_list)
        self.Y_max = self.Y_max_list[0][0]
        for i in range(iterations):
            print("Iteration " + str(i))
            # collects samples from the search space
            samples = self.sample_search_space(X = self.X_max_list, quantity = search_space_sample_size)
            # computes means and variances of search space samples using the surrogate model
            means, variances = self.surrogate_model_inference(samples, mciterations = 8)
            # computes the expected improvement of the surrogate model results
            max_sorted_indicies = self.acquisition_function(means, variances, y_max = self.Y_max, best_count = objective_batch_size)
            # sets self.X_max_list to X values with the largest corresponing Y values
            self.X_max_list = torch.index_select(samples, dim = 0, index = torch.squeeze(max_sorted_indicies))
            # sets self.Y_max_list to the largest Y values
            self.Y_max_list = self.objective(self.X_max_list)
            # adds new data after running objective to the X and Y data
            if i == 0:
                self.X_data = self.X_max_list
                self.Y_data = self.Y_max_list
            else:
                self.X_data = torch.vstack((self.X_data, self.X_max_list))
                self.Y_data = torch.vstack((self.Y_data, self.Y_max_list))

            top_y, nil_index = torch.topk(self.Y_data, k=1, dim=0)
            print("Current Max Y: " + str(top_y.item()))
            self.Y_max = top_y[0][0]
            # updates the surrogate model
            self.surrogate_model_train(self.X_data, self.Y_data)

# Quantum Architecture Search Neural Evolutionary Bayesian Optimization
class QASNEBO(nebo):
    def __init__(self, 
    neural_architecture, 
    qc_gate_options_count, 
    qc_gate_capacity, initial_circuits = None, 
    nn_dropout_p = 0.25):
        
        # initial circut sample if using mutation for sampling the search space
        self.initial_circuits = initial_circuits
        # number of possible gate configuations for a single gate slot
        self.qc_gate_options_count = qc_gate_options_count
        # total number of possible gates in a single circuit
        self.qc_gate_capacity = qc_gate_capacity

        # neural architecture of the NN for the neural predictor
        self.architecture = neural_architecture
        # number of input features
        self.feature_n = self.qc_gate_capacity * self.qc_gate_options_count
        # init neural predictor model
        self.model = Neural_Predictor(
            n_in = 5,#self.feature_n,
            n_out = 1,
            architecture = self.architecture,
            dropout_p=nn_dropout_p)

        self.X_data = None
        self.Y_data = None
        self.X_max_list = None
        self.Y_max_list = None
        self.Y_max = None

    def decode_QCC(self, x):
        gate_encodings = torch.vsplit(x, qc_gate_capacity)
        
        for i in gate_encodings:
            pass

    def objective(self, X):
        # # computes objective function
        # Y = 10 * torch.sin(3.14 * X[:, 0] * X[:, 1]) + 20 * torch.square(X[:, 2] - 0.5) + 10 * X[:, 3] + 5 * X[:, 4]
        # # adds dim at 1 for format
        # Y = torch.unsqueeze(Y, 1)
        # return -Y
        pass

    def sample_search_space(self, X = None, quantity = 1):
        # # sets feature size
        # feature_size = 5
        # # if X is none returns random sample
        # if X == None:
        #     return 20 * (torch.rand(quantity, feature_size)-0.5)
        # # the quantity of samples must be greater than the X input size 
        # # as well as a factor of X input size
        # if X.size(0) > quantity:
        #     raise Exception("canidates to be mutated must be a factor of and less than quantity")
        # scale_size = int(quantity/X.size(0))
        # # returns a mutated output from clones of X inputs
        # output = X.repeat(scale_size, 1) + (2 * (torch.rand(quantity, feature_size) - 0.5))
        # return output
        pass

# Friedman Example with Neural Evolutionary Bayesian Optimization
class ExampleNEBO(nebo):
    def __init__(self, 
    neural_architecture, 
    nn_dropout_p = 0.25):
        # neural architecture of the NN for the neural predictor
        self.architecture = neural_architecture
        # init neural predictor model
        self.model = Neural_Predictor(
            n_in = 5,
            n_out = 1,
            architecture = self.architecture,
            dropout_p=nn_dropout_p)

        self.X_data = None
        self.Y_data = None
        self.X_max_list = None
        self.Y_max_list = None
        self.Y_max = None

    def objective(self, X):
        # computes objective function
        Y = 10 * torch.sin(3.14 * X[:, 0] * X[:, 1]) + 20 * torch.square(X[:, 2] - 0.5) + 10 * X[:, 3] + 5 * X[:, 4]
        # adds dim at 1 for format
        Y = torch.unsqueeze(Y, 1)
        return -Y

    def sample_search_space(self, X = None, quantity = 1):
        # sets feature size
        feature_size = 5
        # if X is none returns random sample
        if X == None:
            return 20 * (torch.rand(quantity, feature_size)-0.5)
        # the quantity of samples must be greater than the X input size 
        # as well as a factor of X input size
        if X.size(0) > quantity:
            raise Exception("canidates to be mutated must be a factor of and less than quantity")
        scale_size = int(quantity/X.size(0))
        # returns a mutated output from clones of X inputs
        output = X.repeat(scale_size, 1) + (2 * (torch.rand(quantity, feature_size) - 0.5))
        return output