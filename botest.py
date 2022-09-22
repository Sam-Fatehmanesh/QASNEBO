import torch
import numpy as np
from QASNEBO.BO.bo import ExampleNEBO, mnistQASNEBO
from QASNEBO.objectives.quantum import tensortocircuit
import matplotlib.pyplot as plt

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

architecture = [
    [3000, "selu"],
    [128, "selu"],
    [64, "selu"],
    [32, "selu"],
    [1, "sigmoid"],
]

n_wires = 10

best_results = []
# for i in range(1):
#     bo_test = mnistQASNEBO(
#         neural_architecture= architecture,
#         qc_n_wires= n_wires,
#         qc_gate_capacity= 50,
#         nn_dropout_p= 0.2,
#         mnist_index=[0,2]
#         )

#     # 10 * 60 * 10 = 6000

#     bo_test.bo_run(
#         iterations= 30,
#         search_space_sample_size= 4096,
#         objective_batch_size= 2,
#         gate_mutation_rate= 0.25,
#         )
#     best_results.append(bo_test.Y_max.item())
#     print(bo_test.Y_max.item())
bo_test = mnistQASNEBO(
    neural_architecture= architecture,
    qc_n_wires= n_wires,
    qc_gate_capacity= 50,
    nn_dropout_p= 0.2,
    mnist_index=[0,2]
    )
bo_test.bo_run(
iterations= 60,
search_space_sample_size= 4096,
objective_batch_size= 4,
gate_mutation_rate= 0.25,
)

# 10 * 60 * 10 = 6000
# try:
#     bo_test.bo_run(
#         iterations= 60,
#         search_space_sample_size= 4096,
#         objective_batch_size= 4,
#         gate_mutation_rate= 0.25,
#         )

# finally:
#     best_results.append(bo_test.Y_max.item())
#     print(bo_test.Y_max.item())

#     pred, true = bo_test.val_surrogate_model(128)
#     mean, var = pred
#     mean = torch.squeeze(mean).cpu().detach().numpy().tolist()
#     true = torch.squeeze(true).cpu().detach().numpy().tolist()
#     print(mean)
#     print(true)

#     plt.scatter(mean, true, c="blue")
#     plt.show()




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