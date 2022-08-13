import os
import botorch as bt
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import ax
import numpy as np
from torch.utils.data import DataLoader, random_split, TensorDataset
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
#import botorch.models.model.Model as bo_base_model
from QASNEBO.BO.bo import ExampleNEBO

architecture = [
    [128, "selu"],
    [128, "selu"],
    [128, "selu"],
    [128, "selu"],
    [128, "selu"],
]

bo_test = ExampleNEBO(architecture, nn_dropout_p=0.2)



bo_test.bo_run(iterations= 20, search_space_sample_size= 100, objective_batch_size= 50)

# X, Y = datasets.make_friedman1(n_samples = 10000,noise = 0.0,random_state = 1,n_features = 5)
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.001, random_state = 42, shuffle = True)


# bo_test.surrogate_model_train(Tensor(X_train), Tensor(Y_train), verbose = True, training_epochs = 1)



#x = torch.rand(1,5)

#print(bo_test.sample_search_space(X=x,quantity=5))

# x = Tensor(X_test)
# mean, var = (bo_test.model.forward_mcd(x, 3))
# print("mean")
# print(mean)





# for i in range(mean.size(dim = 0)):
#     print(i)
#     print(mean)
#     print(var)
# print(Tensor(Y_test))