import torch
from torch import nn
import torch.nn.functional as F
import torchquantum as tq
from torchquantum.datasets import MNIST
import torch.optim as optim

#mnist_index = [1,3,8]

def gen_gate(gate, wires):
    if gate == "rx":
        return tq.RX(has_params=True, trainable=True, wires=wires)
    elif gate == "ry":
        return tq.RY(has_params=True, trainable=True, wires=wires)
    elif gate == "rz":
        return tq.RZ(has_params=True, trainable=True, wires=wires)
    elif gate == "p":
        return tq.PhaseShift(has_params=True, trainable=True, wires=wires)
    elif gate == "cnot":
        return tq.CNOT(wires=wires)
    elif gate == "i":
        return tq.I(wires=wires)
    else:
        raise Exception("Desired gate not part of gate set in use")

def tensortogate(x, n_wires):
    index = (x == 1).nonzero(as_tuple=True)[0].item()
    index += 1

    wire = None
    gate = ""
    if index <= n_wires:
        wire = index - 1
        gate = "rx"
    elif index <= n_wires * 2:
        wire = index - (n_wires * 1) - 1
        gate = "ry"
    elif index <= n_wires * 3:
        wire = index - (n_wires * 2) - 1
        gate = "rz"
    elif index <= n_wires * 4:
        wire = index - (n_wires * 3) - 1
        gate = "p"
    elif index <= n_wires * 5:
        control_index = (x == -1).nonzero(as_tuple=True)[0].item() + 1
        wire = index - (n_wires * 4) - 1
        control_wire = control_index - (n_wires * 4) - 1
        gate = "cnot"
        wire = [wire , control_wire]
    elif index <= n_wires * 6:
        wire = index - (n_wires * 5) - 1
        gate = "i"
    else:
        raise Exception("Error in gate tensor encoding")

    return [gate, wire]

def tensortocircuit(x, n_wires):
    n_gatetypes = 6 # number of single wire gates
    n_gatecombs = (n_wires * n_gatetypes)

    circuit_encodings = []

    for circuit_tensor in x:

        circuit_encoding = []
        encoding_size = list(circuit_tensor.size())[0]

        for i in range(int(encoding_size / n_gatecombs) ):
            start = i * n_gatecombs
            end = (n_gatecombs * (i+1))
            circuit_encoding.append(tensortogate(circuit_tensor[start:end], n_wires))
        
        circuit_encodings.append(circuit_encoding)
    return circuit_encodings    

def gen_MNISTdataflow(mnist_index):
    dataset = MNIST(
    root='./mnist_data',
    train_valid_split_ratio=[0.9, 0.1],
    digits_of_interest=mnist_index,
    n_test_samples=75,
    fashion=True,
    )
    dataflow = dict()

    for split in dataset:
        sampler = torch.utils.data.RandomSampler(dataset[split])
        dataflow[split] = torch.utils.data.DataLoader(
            dataset[split],
            batch_size=256,
            sampler=sampler,
            num_workers=1,
            pin_memory=True,
            multiprocessing_context='fork'
            )

    return dataflow

class QLayer(tq.QuantumModule):
    def __init__(self, n_wires, QArchitecture):
        super().__init__()

        self.n_wires = n_wires

        self.gates = tq.QuantumModuleList()
        for i in QArchitecture:
            self.gates.append(gen_gate(gate=i[0], wires=i[1]))

    
    def forward(self, q_device: tq.QuantumDevice):
        self.q_device = q_device
        for gate in self.gates:
            gate(self.q_device)

class mnistVQA(tq.QuantumModule):
    def __init__(self, QArchitecture, n_wires, mnist_index):
        super().__init__()
        self.n_wires = n_wires
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
    
        self.encoder = tq.AmplitudeEncoder()

        

        self.q_layer = QLayer(self.n_wires, QArchitecture)
        self.measure = tq.MeasureAll(tq.PauliZ)



        #Classical Stack
        self.preQstack = nn.Sequential(
            nn.ReflectionPad2d(2),
            nn.Flatten(),
        )
        self.postQstack = nn.Sequential(
            nn.Linear(self.n_wires, len(mnist_index)),
            nn.LogSoftmax(dim= 1)
        )

    def forward(self, x):

        x = self.preQstack(x)
        self.encoder(self.q_device, x)
    
        self.q_layer(self.q_device)
        x = self.measure(self.q_device)

        x = self.postQstack(x)

        return x

def train_batch(dataflow, model, device, optimizer):
    for feed_dict in dataflow['train']:
        inputs = feed_dict['image'].to(device)
        targets = feed_dict['digit'].to(device)

        outputs = model(inputs)
        loss = F.nll_loss(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def valid_test(dataflow, split, model, device):
    target_all = []
    output_all = []
    with torch.no_grad():
        for feed_dict in dataflow[split]:
            inputs = feed_dict['image'].to(device)
            targets = feed_dict['digit'].to(device)

            outputs = model(inputs)

            target_all.append(targets)
            output_all.append(outputs)
        target_all = torch.cat(target_all, dim=0)
        output_all = torch.cat(output_all, dim=0)

    _, indices = output_all.topk(1, dim=1)
    masks = indices.eq(target_all.view(-1, 1).expand_as(indices))
    size = target_all.shape[0]
    corrects = masks.sum().item()
    accuracy = corrects / size
    loss = F.nll_loss(output_all, target_all).item()


    return accuracy

def train(dataflow, model, device, epochs=5):
    optimizer = optim.Adam(model.parameters(), lr=5e-3, weight_decay=1e-4)

    for epoch in range(1, epochs + 1):
        # train
        train_batch(dataflow, model, device, optimizer)
    
    return valid_test(dataflow, 'test', model, device)
