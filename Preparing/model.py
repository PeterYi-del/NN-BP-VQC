import numpy as np
import torch.nn as nn
import functools
import math

import tensorcircuit as tc
import torch

from noise_parameter import single_qubit_dep_channel, two_qubit_dep_channel


# NEQC with Neural Networks
class NEQC_NN(nn.Module):
    def __init__(self, num_qubits, num_blocks, circuit, input_state, circuit_state):
        super(NEQC_NN, self).__init__()
        self.num_qubits = num_qubits  # number of qubits
        self.num_blocks = num_blocks  # number of blocks
        self.circuit = circuit  # quantum circuit
        self.input_state = input_state  # random input quantum state
        self.circuit_state = circuit_state  # quantum circuit state 'pure' or 'noise'

        # connect tensorcircuit to torch and enable JIT compilation
        self.circuit_eval_probs = tc.interfaces.torch_interface(self.circuit_eval_probs, jit=True)

        # initialize alpha
        init_method = functools.partial(torch.nn.init.uniform_, b=2 * math.pi)
        self.alpha = torch.nn.Parameter(init_method(torch.Tensor(1, 4)))

        # define fully connected layers in a neural network
        self.l1 = nn.Linear(4, 10)
        self.l2 = nn.Linear(10, 20)
        self.l3 = nn.Linear(20, (3 * self.num_qubits + 4 * self.num_blocks))
        self.func = nn.PReLU()  # activation Function
        self.layernorm1 = nn.LayerNorm(10)
        self.layernorm2 = nn.LayerNorm(20)
        self.layernorm3 = nn.LayerNorm(3 * self.num_qubits + 4 * self.num_blocks)

    def forward(self, alpha, scheme):
        if scheme == 'neqc_express':  # the express process uses external-alpha
            y = self.l1(alpha)
        else:
            y = self.l1(self.alpha)  # the training process uses self-alpha
        y = self.layernorm1(y)
        y = self.func(y)
        y = self.l2(y)
        y = self.layernorm2(y)
        y = self.func(y)
        y = self.l3(y)
        y = self.layernorm3(y)
        y = self.func(y)

        # tailoring output shape to match quantum circuit parameters
        y = y.reshape((3 * self.num_qubits + 4 * self.num_blocks))

        # returns the corresponding output according to the scheme
        if scheme == 'neqc_express':
            return y  # the express process return trained parameters
        else:
            probs = self.circuit_eval_probs(y)  # the training process return quantum state probability
            return probs

    # calculating probability
    def circuit_eval_probs(self, weight):
        qc = tc.Circuit(self.num_qubits, inputs=self.input_state)
        blocks = self.circuit.to_qir()
        label = 0  # used to keep track of the number of 'cz' gates skipped

        for i, block in enumerate(blocks):
            gate = block['name']
            qubits = block['index']

            if gate in ['ry', 'rz']:
                # for the 'ry' and 'rz' gates, we need to adjust the index to skip the 'cz' gate
                adjusted_index = i - label
                theta = weight[adjusted_index]
                getattr(qc, gate)(qubits[0], theta=theta)  # use getattr to dynamically call gates
                if self.circuit_state == 'noise':
                    if gate == 'ry':
                        qc.general_kraus(single_qubit_dep_channel, qubits[0])
                    elif gate == 'rz':
                        qc.general_kraus(single_qubit_dep_channel, qubits[0])
            elif gate == 'cz':
                # for the 'cz' gate, we apply the gate directly and increment the label count
                qc.cz(*qubits)
                label += 1
                if self.circuit_state == 'noise':
                    qc.general_kraus(two_qubit_dep_channel, *qubits)

        return qc.probability()


# 卷积神经网络
class NEQC_CNN(nn.Module):
    def __init__(self, num_qubits, num_blocks, circuit, input_state, circuit_state):
        super(NEQC_CNN, self).__init__()
        self.num_qubits = num_qubits  # number of qubits
        self.num_blocks = num_blocks  # number of blocks
        self.circuit = circuit  # quantum circuit
        self.input_state = input_state  # random input quantum state
        self.circuit_state = circuit_state  # quantum circuit state 'pure' or 'noise'

        # connect tensorcircuit to torch and enable JIT compilation
        self.circuit_eval_probs = tc.interfaces.torch_interface(self.circuit_eval_probs, jit=True)

        self.alpha = torch.rand(1, 1, 4) * (2 * math.pi)  # initial alpha

        out_channels = math.ceil((
                                         3 * self.num_qubits + 4 * self.num_blocks) // 4) + 1  # This setting is
        # to ensure that the parameters after CNN training are close to the total number required

        self.conv1 = nn.Conv1d(in_channels=1, out_channels=10, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=10, out_channels=20, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(in_channels=20, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.act = nn.PReLU()

    def forward(self, alpha, scheme):
        if scheme == 'neqc_express':
            y = self.conv1(alpha)  # the express process uses external-alpha
        else:
            y = self.conv1(self.alpha)  # the training process uses self-alpha
        y = self.act(y)
        y = self.conv2(y)
        y = self.act(y)
        y = self.conv3(y)
        y = y.view(-1)
        num_features = y.numel()  # get the number of y
        norm = nn.LayerNorm(num_features)  # layer norm
        y = norm(y)
        y = y[:3 * self.num_qubits + 4 * self.num_blocks]  # select the number of parameters we need
        y = y.reshape((3 * self.num_qubits + 4 * self.num_blocks))

        # returns the corresponding output according to the scheme
        if scheme == 'neqc_express':
            return y  # the express process return trained parameters
        else:
            probs = self.circuit_eval_probs(y)  # the training process return quantum state probability
            return probs

    # evaluate probability
    def circuit_eval_probs(self, weight):
        qc = tc.Circuit(self.num_qubits, inputs=self.input_state)
        blocks = self.circuit.to_qir()
        label = 0  # used to keep track of the number of 'cz' gates skipped

        for i, block in enumerate(blocks):
            gate = block['name']
            qubits = block['index']

            if gate in ['ry', 'rz']:
                # for the 'ry' and 'rz' gates, we need to adjust the index to skip the 'cz' gate
                adjusted_index = i - label
                theta = weight[adjusted_index]
                getattr(qc, gate)(qubits[0], theta=theta)  # use getattr to dynamically call gates
                if self.circuit_state == 'noise':
                    if gate == 'ry':
                        qc.general_kraus(single_qubit_dep_channel, qubits[0])
                    elif gate == 'rz':
                        qc.general_kraus(single_qubit_dep_channel, qubits[0])
            elif gate == 'cz':
                # for the 'cz' gate, we apply the gate directly and increment the label count
                qc.cz(*qubits)
                label += 1
                if self.circuit_state == 'noise':
                    qc.general_kraus(two_qubit_dep_channel, *qubits)

        return qc.probability()


# SQC without Neural Network
class SQC(nn.Module):
    def __init__(self, num_qubits, num_blocks, circuit, input_state, circuit_state):
        super(SQC, self).__init__()
        self.num_qubits = num_qubits  # number of qubits
        self.num_blocks = num_blocks  # number of blocks
        self.circuit = circuit  # quantum circuit
        self.input_state = input_state  # random input quantum state
        self.circuit_state = circuit_state  # quantum circuit state 'pure' or 'noise'

        # connect tensorcircuit to torch and enable JIT compilation
        self.circuit_eval_probs = tc.interfaces.torch_interface(self.circuit_eval_probs, jit=True)

        # initialize the parameters required by the circuit using uniform distribution
        init_method = functools.partial(torch.nn.init.uniform_, b=2 * math.pi)
        self.alpha = torch.nn.Parameter(init_method(torch.Tensor(3 * self.num_qubits + 4 * self.num_blocks)),
                                        requires_grad=True)

    def forward(self, alpha, scheme):
        probs = self.circuit_eval_probs(self.alpha)  # return quantum state probability
        return probs

    # evaluate probability
    def circuit_eval_probs(self, weight):
        qc = tc.Circuit(self.num_qubits, inputs=self.input_state)
        blocks = self.circuit.to_qir()
        label = 0  # used to keep track of the number of 'cz' gates skipped

        for i, block in enumerate(blocks):
            gate = block['name']
            qubits = block['index']

            if gate in ['ry', 'rz']:
                # for the 'ry' and 'rz' gates, we need to adjust the index to skip the 'cz' gate
                adjusted_index = i - label
                theta = weight[adjusted_index]
                getattr(qc, gate)(qubits[0], theta=theta)  # use getattr to dynamically call gates
                if self.circuit_state == 'noise':
                    if gate == 'ry':
                        qc.general_kraus(single_qubit_dep_channel, qubits[0])
                    elif gate == 'rz':
                        qc.general_kraus(single_qubit_dep_channel, qubits[0])
            elif gate == 'cz':
                # for the 'cz' gate, we apply the gate directly and increment the label count
                qc.cz(*qubits)
                label += 1
                if self.circuit_state == 'noise':
                    qc.general_kraus(two_qubit_dep_channel, *qubits)

        return qc.probability()
