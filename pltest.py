import datetime
import torch
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
from joblib import Parallel, delayed
#writer = SummaryWriter()

# compute device management
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

def actfunc_from_str(actfuncstring):
    if actfuncstring == "sigmoid":
        return nn.Sigmoid()
    elif actfuncstring == "selu":
        return nn.SELU()
    elif actfuncstring == "relu":
        return nn.ReLU()
    elif actfuncstring == "tanh":
        return nn.Tanh()
    else:
        raise Exception("ERROR: Requested activation function does not exist")

# init weights linearly in combination for self normalizing 
def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.calculate_gain("linear", m.weight)

# functions to apply to nn.Sequencial in order to turn on/off dropout for MCD
def activate_dropout(m):
    if isinstance(m, nn.Dropout):
        m.train(True)

def deactivate_dropout(m):
    if isinstance(m, nn.Dropout):
        m.train(False)

# model definition
class NN(pl.LightningModule):
    #_num_outputs = 1
    def __init__(self, n_in, n_out, architecture,  dropout_p = 0.0, log_torchmetric = False):
        super(NN, self).__init__()

        # metric logging
        #self.accuracy = torchmetrics.Accuracy()
        self.log_torchmetric = log_torchmetric

        # set loss function
        self.loss_fn = nn.MSELoss()

        # the drop out probability
        self.dropout_p = dropout_p

        # init the main model stack
        self.stack = nn.Sequential()
        # dynamic model construction
        self.stack.append(nn.AlphaDropout(dropout_p))
        for i, neural_layer_info in enumerate(architecture):
            neural_width = neural_layer_info[0]
            act_func = actfunc_from_str(neural_layer_info[1])

            if i == 0:
                self.stack.append(nn.Linear(n_in, neural_width))
                self.stack.append(act_func)
                self.stack.append(nn.AlphaDropout(dropout_p))

            else:
                self.stack.append(nn.Linear(neural_width, neural_width))
                self.stack.append(act_func)
                self.stack.append(nn.AlphaDropout(dropout_p))
                    
        self.stack.append(nn.Linear(neural_width, n_out))

        # init weights
        self.stack.apply(init_weights)
        
    def forward(self, x):
        return self.stack(x)

    # monto carlo dropout foward - returns the output mean and variance
    def forward_mcd(self, x, mciterations):
        # activates dropout
        self.stack.apply(activate_dropout)
        # collects mciterations outputs from the neural network and places in torch.vstack
        outputs = torch.vstack([self.forward(x) for i in range(mciterations)])
        # deactivates dropout
        self.stack.apply(deactivate_dropout)

        mean = torch.mean(input = outputs, dim = 0)
        variance = torch.var(input = outputs, dim = 0)

        return mean, variance

    def configure_optimizers(self):
       return torch.optim.Adam(self.parameters(), 0.001)

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        out = self.forward(x)
        loss = self.loss_fn(out, y)
        if self.log_torchmetric:
            self.log("train_loss", loss)
        return loss

    def validation_step(self, valid_batch, batch_idx):
        x, y = valid_batch
        out = self.forward(x)
        loss = self.loss_fn(out, y)
        if self.log_torchmetric:
            self.log("val_loss", loss)
        return loss


# friedman datagen
X, Y = datasets.make_friedman1(n_samples = 10000,noise = 0.0,random_state = 1,n_features = 10)

# test datagen
#X = np.random.rand(3000*20,2) * 5
#Y = 10 * ((np.sin(X[:,0] * 3) + 20) + np.power(X[:,0],0.5))

# splits dataset into training and testing data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.1, random_state = 42, shuffle = True)

# configures dataloaders and turns numpy arrays into torch tensors
train_dl1 = DataLoader(TensorDataset(torch.Tensor(X_train), torch.Tensor(Y_train)), batch_size = 1, num_workers = 2, multiprocessing_context='fork')
#test_dl = DataLoader(TensorDataset(torch.Tensor(X_test), torch.Tensor(Y_test)), batch_size = 1, num_workers = 6)
train_dl2 = DataLoader(TensorDataset(torch.Tensor(X_train), torch.Tensor(Y_train)), batch_size = 1, num_workers = 2, multiprocessing_context='fork')

# define architecture
architecture = [
    [256, "selu"],
    [256, "selu"],
    [256, "selu"],
    [256, "selu"],
    [256, "selu"],
]

# init model
model1 = NN(10, 1, architecture = architecture, dropout_p = 0.1)
model2 = NN(10, 1, architecture = architecture, dropout_p = 0.1)


# init trainer
trainer1 = pl.Trainer(max_epochs = 2, accelerator = 'gpu', devices = 1)
trainer2 = pl.Trainer(max_epochs = 2, accelerator = 'gpu', devices = 1)

# train/validate the model

trainers = [trainer1, trainer2]
models = [model1, model2]

traindatas = [train_dl1, train_dl2]
print("OooooooooOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOooooooooooOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO")

Parallel(n_jobs=2)(delayed(trainers[i].fit)(models[i], traindatas[i]) for i in range(len(trainers)))

print("#######################################################################")






#trainer.validate(model, test_dl)

# saves model
# torch.save(model.state_dict(), 'models/' + str(datetime.datetime.now()) + '.pth')

# # generating some examples for evaluation of model training
# print("Running inference...")
# for i in range(4):
#     print("predicted mean & variance:")
#     mean, var = model.forward_mcd(torch.Tensor(X_test[i]), mciterations = 6)
#     print(mean.cpu().detach().numpy())
#     print(var.cpu().detach().numpy())
#     #print(model(torch.Tensor(X_test[i])).cpu().detach().numpy())
#     print("correct mean:")
#     print(Y_test[i])

# print("Running inference...")
# # test data output for validation checking
# yval_out = np.zeros(0)
# for i in range(len(X_test)):
#     yval_out = np.append(yval_out, model(torch.Tensor(X_test[i])).cpu().detach().numpy())

# # train data output to check for overfit
# n_points = int(len(X_train[:,0]) * 11)
# Y_out = np.zeros(0)
# for i in range(n_points):
#     Y_out = np.append(Y_out, model(torch.Tensor(X_train[i])).cpu().detach().numpy())

# print("Generating plot graphics...")

# fig = plt.figure()
# ax1 = fig.add_subplot(111)

# ax1.scatter(X[0:n_points,0], Y[0:n_points],c="b",label="correct") # plotting true correct data
# ax1.scatter(X_test[:,0],yval_out,c="r",label="predictions") # plotting test data predictions
# ax1.scatter(X_train[0:n_points,0],Y_out,c="g",label="train_data_predictions") # plotting train data predictions

# plt.legend(loc='upper left')
# plt.show()

# notes
# - low dimentionality and high batch size may cause averaging and the NN outputting 