from math import factorial
import torch
from torch import nn, Tensor
import torchquantum as tq


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
    else:
        raise Exception("Desired gate not part of gate set in use")

def tensortogate(x, n_wires):
    n_gatetype_one = 4 # number of single wire gates
    n_gatetypes = (n_wires * n_gatetype_one)

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
    else:
        raise Exception("Error in gate tensor encoding")

    return [gate, wire]

def tensortocircuit(x, n_wires):

    n_gatetypes = 5 # number of single wire gates
    n_gatecombs = (n_wires * n_gatetypes)

    circuit_encodings = []

    for circuit_tensor in x:
        circuit_encoding = []
        encoding_size = list(circuit_tensor.size())[0]
        # print(int(encoding_size / n_wires))
        for i in range(int(encoding_size / n_gatecombs) ):
            start = i * n_gatecombs
            end = (n_gatecombs * (i+1))
            circuit_encoding.append(tensortogate(circuit_tensor[start:end], n_wires))
        
        circuit_encodings.append(circuit_encoding)
    return circuit_encodings



