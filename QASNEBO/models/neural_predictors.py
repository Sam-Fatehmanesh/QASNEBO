import torch
import torch.nn as nn
import pytorch_lightning as pl

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
        torch.nn.init.calculate_gain('linear', m.weight)

# functions to apply to nn.Sequencial in order to turn on/off dropout for MCD
def activate_dropout(m):
    if isinstance(m, nn.Dropout):
        m.train(True)

def deactivate_dropout(m):
    if isinstance(m, nn.Dropout):
        m.train(False)

# model definition
class Neural_Predictor(pl.LightningModule):
    #_num_outputs = 1
    def __init__(self, n_in, n_out, architecture,  dropout_p = 0.0, log_torchmetric = False):
        super(Neural_Predictor, self).__init__()

        # metric logging
        self.accuracy = None
        
        if log_torchmetric:
            self.accuracy = torchmetrics.Accuracy()

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

    def forward_d(self, x, iterations):
        # activates dropout
        self.stack.apply(activate_dropout)
        # collects mciterations outputs from the neural network and places in torch.vstack
        outputs = torch.vstack([torch.unsqueeze(self.forward(x), dim = 0) for i in range(iterations)])
        # deactivates dropout
        self.stack.apply(deactivate_dropout)

        return outputs

    # monto carlo dropout foward - returns the output mean and variance
    def forward_mcd(self, x, mciterations):
        outputs = self.forward_d(x, mciterations)
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

    def reset_weights(self):
        self.stack.apply(init_weights)