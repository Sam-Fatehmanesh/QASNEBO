import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
from torch.utils.data import DataLoader, random_split, TensorDataset
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from QASNEBO.models.neural_predictors import NeuralPredictor
from torch import Tensor
from torch.distributions.normal import Normal
from QASNEBO.objectives.quantum import tensortocircuit, mnistVQA, gen_MNISTdataflow
from QASNEBO.objectives import quantum
from joblib import Parallel, delayed
import psutil

class BayesOpt:
    def surrogate_model_train(self, x_train: Tensor, y_train: Tensor, x_pred: Tensor):
        
        pass

    def surrogate_model_inference(self, x: Tensor):
        
        raise NotImplementedError

    def objective(self, x: Tensor):
        
        raise NotImplementedError
    
    def acquisition_function(self, x: Tensor):
        
        raise NotImplementedError

    def sample_objective_domain(self):

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
    training_epochs = 6, 
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
        #trainer.fit(self.model, train_dataloaders = train_dl)
    
    def surrogate_model_inference(self, x_pred, mciterations):
        # returns model inference output in the form of mean, variances
        mean, variances = self.model.forward_mcd(x_pred, mciterations = mciterations)

        return mean, variances

    def acquisition_function(self, means, variances, y_max, best_count = 1):
        # expected improvement constant
        ei_constant = 0.5

        # converts var to std and multiply by a calibration constant
        stds = torch.sqrt(variances) * ei_constant

        # normal distribution
        dist = Normal(means, stds)
        # Z score
        Ymax = y_max
        Z = (Ymax - (means)) * stds.reciprocal()
        # cumulative distributions
        cd = dist.cdf(torch.squeeze(Ymax))
        # probability densities
        pd = dist.log_prob(Ymax).exp()
        # computes expected improvements
        ei = (-stds) * (Z * cd + pd)
        # sorted ei indices
        ei_sorted, ei_sorted_indices = torch.topk(input = ei, dim = 0, k = best_count)

        # returns expected improvements and their corresponding sorted indices
        return ei_sorted_indices

    def bo_run(self, iterations, search_space_sample_size, objective_batch_size = 1, gate_mutation_rate = 0.1):
        if search_space_sample_size % objective_batch_size != 0:
            raise Exception("search_space_sample_size must be divisible by objective_batch_size")
        # collects a sample from the search space
        base_ansatz = torch.unsqueeze(torch.zeros(self.feature_n),dim=0)
        self.X_max_list = self.sample_objective_domain(X = base_ansatz, quantity = 1, gate_mutation_rate = 1)
        # computes the objective value of a single sample to serve as the first maximum
        print("performing first run of objective func")
        self.Y_max_list = self.objective(self.X_max_list)
        print("done")
        #print(self.Y_max_list)
        self.Y_max = self.Y_max_list[0]

        old_Y_max_avg = torch.mean(self.Y_max_list)
        gen_circuits = self.X_max_list

        for i in range(iterations):
            print("Iteration " + str(i + 1))
            # collects samples from the search space




            print("collecting samples")
            samples = self.sample_objective_domain(
                X = gen_circuits, 
                quantity = search_space_sample_size, 
                gate_mutation_rate = gate_mutation_rate
                )
            print("done")
            # computes means and variances of search space samples using the surrogate model
            print("inferencing surrogate model")
            means, variances = self.surrogate_model_inference(samples, mciterations = 8)
            print("done")
            # computes the expected improvement of the surrogate model results
            print("computing acquisition function")
            max_sorted_indices = self.acquisition_function(means, variances, y_max = self.Y_max, best_count = objective_batch_size)
            print("done")
            # sets self.X_max_list to X values with the largest corresponding Y values
            self.X_max_list = torch.index_select(samples, dim = 0, index = torch.squeeze(max_sorted_indices))
            # sets self.Y_max_list to the largest Y values
            print("computing objective function")
            self.Y_max_list = self.objective(self.X_max_list)
            print("done")
            # adds new data after running objective to the X and Y data
            if i == 0:
                self.X_data = self.X_max_list
                self.Y_data = self.Y_max_list
            else:
                self.X_data = torch.cat((self.X_data, self.X_max_list))
                self.Y_data = torch.cat((self.Y_data, self.Y_max_list))
            #print(self.Y_data)
            #print(self.Y_data)
            top_y = torch.max(self.Y_data)
            top_x_index = torch.argmax(self.Y_data, dim= 0)
            #print(self.Y_data)
            #print(top_y)
            print("Current Max Y: " + str(top_y.item()))
            
            if self.Y_max < top_y:
                print("updated Y max")
                self.Y_max = top_y
                self.X_max = self.X_data[top_x_index.item()]


            new_Y_max_avg = torch.mean(self.Y_max_list)
            if old_Y_max_avg < new_Y_max_avg:
                print("genesis circuits updated")
                gen_circuits = self.X_max_list
                old_Y_max_avg = new_Y_max_avg
            # updates the surrogate model
            if i < iterations - 1:
                print("training surrogate model")
                self.surrogate_model_train(self.X_data, self.Y_data)
                print("done")

    def val_surrogate_model(self, count):
        base_ansatz = torch.unsqueeze(torch.zeros(self.feature_n),dim=0)
        test_Xs = self.sample_objective_domain(X = base_ansatz, quantity = count, gate_mutation_rate = 1)
        pred_val_accs = self.surrogate_model_inference(test_Xs, 8)
        true_val_accs = torch.vstack([self.objective(torch.unsqueeze(i, dim=0)) for i in test_Xs])
        return pred_val_accs, true_val_accs


# Quantum Architecture Search Neural Evolutionary Bayesian Optimization
class mnistQASNEBO(nebo):
    def __init__(self, 
    neural_architecture, 
    qc_n_wires, 
    qc_gate_capacity, 
    initial_circuits = None, 
    vqa_train_epochs = 2,
    mnist_index = [0,2],
    nn_dropout_p = 0.25,):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("cuda" if torch.cuda.is_available() else "cpu")
        # initial circuit sample if using mutation for sampling the search space
        self.initial_circuits = initial_circuits
        # number of possible gate configurations for a single gate slot
        self.qc_gate_options_count = 6
        # total number of possible gates in a single circuit
        self.qc_gate_capacity = qc_gate_capacity

        self.qc_n_wires = qc_n_wires

        # neural architecture of the NN for the neural predictor
        self.architecture = neural_architecture
        # number of input features
        self.feature_n = self.qc_gate_capacity * self.qc_gate_options_count * self.qc_n_wires
        # init neural predictor model
        self.model = NeuralPredictor(
            n_in = self.feature_n,
            n_out = 1,
            architecture = self.architecture,
            dropout_p=nn_dropout_p)

        self.vqa_train_epochs = vqa_train_epochs
        self.mnist_index = mnist_index

        self.X_data = None
        self.Y_data = None
        self.X_max_list = None
        self.Y_max_list = None
        self.Y_max = None
        self.X_max = None

    # runs objective function with quantum ansatz encoding as input 
    # returning accuracy
    def objective(self, X):
        
        # convert encoded circuit batch into string/int representations
        decoded_circuit_defs = tensortocircuit(X, self.qc_n_wires)
        # the number of quantum algos to test
        n_QVAs = len(decoded_circuit_defs)

        generated_QVAs = []
        for decoded_circuit in decoded_circuit_defs:
            # append generated quantum variational algorithms
            generated_QVAs.append(mnistVQA(decoded_circuit, self.qc_n_wires, self.mnist_index).to(self.device))

        dataflows = []
        for i in range(n_QVAs):
            dataflows.append(gen_MNISTdataflow(self.mnist_index))

        # trains and returns accuracies of different quantum variational
        # algos in parallel
        current_process = psutil.Process()
        subproc_before = set([p.pid for p in current_process.children(recursive=True)])
        accuracies = Parallel(n_jobs=n_QVAs)(delayed(quantum.train)(dataflows[i], generated_QVAs[i], self.device, epochs=self.vqa_train_epochs) for i in range(n_QVAs))
        subproc_after = set([p.pid for p in current_process.children(recursive=True)])
        for subproc in subproc_after - subproc_before:
            print('Killing process with pid {}'.format(subproc))
            psutil.Process(subproc).terminate()



        del generated_QVAs
        del dataflows

        return Tensor(accuracies)
    
    # mutation sample search space function
    def sample_objective_domain(self, X = None, quantity = 1, gate_mutation_rate = 0.1):
        gate_encode_size = self.qc_n_wires*self.qc_gate_options_count
        X_neo = X
        n_children = int(quantity / X.size(dim=0))
        #print("children per "+str(n_children))
        all_children = []
        for parent in X_neo:
            children = []
            
            for child in range(n_children):
                #print("child #" + str(child))
                anatz_clone = parent.clone()

                # mutation loop for each gate
                for i in range(self.qc_gate_capacity):

                    mutate = torch.rand(1)

                    if mutate[0].item() < gate_mutation_rate:
                        # make random tensor representing each gate
                        random_tensor = torch.randint(0, gate_encode_size, (1,))
                
                        # convert random tensor into one hot encoding representing each non control gate
                        gate = F.one_hot(random_tensor, num_classes= gate_encode_size)
                        gate = torch.squeeze(gate)
            
                        index = random_tensor.item()
    

                        if index < self.qc_n_wires * 5 and self.qc_n_wires * 4 <= index:
                            #print("adding control for cnot")
                            start = (self.qc_n_wires * 4) 
                            end = (self.qc_n_wires * 5) 
                            cnot_gate = gate[start : end]
                            cnot_gate_index = index - (self.qc_n_wires * 4) - 1
                            random_control_pos = torch.randint(0, self.qc_n_wires-1, (1,))
                            random_control_term = -1 * F.one_hot(random_control_pos, num_classes= self.qc_n_wires-1)
                            #print("random_control_term")
                            #print(random_control_term)
                            random_control_term = random_control_term[0] #torch.squeeze(random_control_term)
                            
                            #print(random_control_term)
                            cnot_gate = torch.cat([
                                random_control_term[:cnot_gate_index],
                                torch.ones(1),
                                random_control_term[cnot_gate_index:]
                            ])
                            gate[start : end] = cnot_gate
                            #print("cnot_gate")
                            #print(cnot_gate)
    
                        anatz_clone[i * gate_encode_size:(i+1) * gate_encode_size] = gate
            
                children.append(anatz_clone)
            all_children.append(torch.vstack(children))


        return torch.vstack(all_children)          
            
# Friedman Example with Neural Evolutionary Bayesian Optimization
class ExampleNEBO(nebo):
    def __init__(self, 
    neural_architecture, 
    nn_dropout_p = 0.25):
        # neural architecture of the NN for the neural predictor
        self.architecture = neural_architecture
        # init neural predictor model
        self.model = NeuralPredictor(
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

    def sample_objective_domain(self, X = None, quantity = 1):
        # sets feature size
        feature_size = 5
        # if X is none returns random sample
        if X == None:
            return 20 * (torch.rand(quantity, feature_size)-0.5)
        # the quantity of samples must be greater than the X input size 
        # as well as a factor of X input size
        if X.size(0) > quantity:
            raise Exception("candidates to be mutated must be a factor of and less than quantity")
        scale_size = int(quantity/X.size(0))
        # returns a mutated output from clones of X inputs
        output = X.repeat(scale_size, 1) + (2 * (torch.rand(quantity, feature_size) - 0.5))
        return output