import functools
import math
import os

import numpy as np
import tensorcircuit as tc
import tensorflow as tf
import torch
from matplotlib import pyplot as plt
from scipy import stats
from noise_parameter import single_qubit_dep_channel, two_qubit_dep_channel

K = tc.set_backend('tensorflow')


class computeNeqcExpress:
    def __init__(self, num_bins, num_qubits, num_fidelity, circuit, circuit_state):
        self.num_bins = num_bins  # number of bins
        self.num_qubits = num_qubits  # number of qubits
        self.circuit = circuit  # quantum circuit
        self.num_fidelity = num_fidelity  # number of fidelities
        self.circuit_state = circuit_state  # circuit state 'pure' or 'noise'

        # compute p_fidelity_Haar
        interval = 1 / self.num_bins  # the interval between each bin
        bins_list = [interval * i for i in range(num_bins + 1)]  # create a list of bins
        result = []
        for i in range(1, len(bins_list)):
            # compute the contribution of each bin, using the Haar measure
            # temp1: the value for the current bin
            temp1 = -1 * np.power((1 - bins_list[i]), np.power(2, self.num_qubits) - 1)
            # temp0: the value for the previous bin
            temp0 = -1 * np.power((1 - bins_list[i - 1]), np.power(2, self.num_qubits) - 1)
            # append the difference (p_fidelity) to the result list
            result.append(temp1 - temp0)
        self.p_fidelity_Haar = np.array(result)  # convert the result list into a numpy array

    def compute_fidelity(self, state1, state2):
        # calculate the inner product between the two states (state1 and state2)
        fidelity = (state1 * state2.conjugate()).sum(
            -1)  # compute the sum of the element-wise product of state1 and the complex conjugate of state2

        # take the absolute value of the inner product and then square it to get the fidelity
        fidelity = np.power(np.absolute(fidelity), 2)  # Square the magnitude of the inner product

        return fidelity

    def compute_neqc_express(self, model, model_name):
        # use JIT-optimized vmap function for parallel evaluation of circuits
        vmap_cir_eval = tc.backend.jit(tc.backend.vmap(self.circuit_eval_state))

        params_array = []  # list to store the circuit parameters obtained from the fixed neural network

        model.eval()  # set the model to evaluation mode, freezing the model's parameters
        # perform inference without updating the model weights
        with torch.no_grad():
            for i in range(2 * self.num_fidelity):
                # initialize the parameter alpha, ensuring requires_grad=False so it won't be updated
                if model_name == 'NEQC-NN':
                    init_method = functools.partial(torch.nn.init.uniform_, b=2 * math.pi)
                    # create a parameter alpha with random initialization in the range [0, 2π]
                    alpha = torch.nn.Parameter(init_method(torch.Tensor(1, 4)), requires_grad=False)
                elif model_name == 'NEQC-CNN':
                    # for CNN, initialize alpha as random values between 0 and 2π
                    alpha = torch.rand(1, 1, 4) * (2 * math.pi)

                # append the result of the model with the parameter alpha for each inference step
                params_array.append(model(alpha, 'neqc_express'))

        # convert the list of parameters to a numpy array
        samples_array = np.array(params_array)

        # convert the samples_array into a form usable by vmap (no change in data values)
        param = tf.Variable(samples_array)

        # compute the output states of the quantum circuits using the vmap function
        output_state_all = vmap_cir_eval(param)
        output_state_all = output_state_all.numpy()

        # split the output states into two parts: output_states1 and output_states2
        output_states1 = output_state_all[0:int(self.num_fidelity)]
        output_states2 = output_state_all[int(self.num_fidelity):]

        # compute fidelities between the two sets of output states
        fidelities = self.compute_fidelity(output_states1, output_states2)  # Returns fidelities

        # map the fidelities to the corresponding bins
        bin_index = np.floor(fidelities * self.num_bins).astype(int)

        # count the number of fidelities in each bin
        num = []
        for i in range(0, self.num_bins):
            num.append(len(bin_index[bin_index == i]))

        # compute the probability distribution over bins
        p_fidelity = np.array(num) / sum(num)

        # compute the Kullback-Leibler (K-L) divergence between the computed p_fidelity and the Haar measure
        express = stats.entropy(p_fidelity, self.p_fidelity_Haar)

        # Clear the parameters-array to free memory
        del params_array

        return express

    # evaluate state
    def circuit_eval_state(self, weight):
        qc = tc.Circuit(self.num_qubits)
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

        return qc.state()


class computeSqcExpress:
    def __init__(self, num_bins, num_qubits, num_fidelity, circuit, circuit_state):
        self.num_bins = num_bins  # number of bins
        self.num_qubits = num_qubits  # number of qubits
        self.circuit = circuit  # quantum circuit
        self.num_fidelity = num_fidelity  # number of fidelities
        self.circuit_state = circuit_state  # circuit state 'pure' or 'noise'

        # compute p_fidelity_Haar
        interval = 1 / self.num_bins  # the interval between each bin
        bins_list = [interval * i for i in range(num_bins + 1)]  # create a list of bins
        result = []
        for i in range(1, len(bins_list)):
            # compute the contribution of each bin, using the Haar measure
            # temp1: the value for the current bin
            temp1 = -1 * np.power((1 - bins_list[i]), np.power(2, self.num_qubits) - 1)
            # temp0: the value for the previous bin
            temp0 = -1 * np.power((1 - bins_list[i - 1]), np.power(2, self.num_qubits) - 1)
            # append the difference (p_fidelity) to the result list
            result.append(temp1 - temp0)
        self.p_fidelity_Haar = np.array(result)  # convert the result list into a numpy array

    def compute_fidelity(self, state1, state2):
        # calculate the inner product between the two states (state1 and state2)
        fidelity = (state1 * state2.conjugate()).sum(
            -1)  # compute the sum of the element-wise product of state1 and the complex conjugate of state2

        # take the absolute value of the inner product and then square it to get the fidelity
        fidelity = np.power(np.absolute(fidelity), 2)  # Square the magnitude of the inner product

        return fidelity

    def compute_sqc_express(self, num_block):
        # use JIT-optimized vmap for parallelized circuit evaluation
        vmap_cir_eval = tc.backend.jit(tc.backend.vmap(self.circuit_eval_state))

        # Randomly sample parameters between 0 and 2π
        param = tf.Variable(
            np.random.uniform(low=0, high=2 * np.pi,
                              size=[self.num_fidelity * 2, (4 * num_block + 3 * self.num_qubits)])
        )

        # evaluate the quantum circuit with the sampled parameters using vmap
        output_state_all = vmap_cir_eval(param)
        output_state_all = output_state_all.numpy()

        # split the output states into two parts for fidelity comparison
        output_states1 = output_state_all[0:int(self.num_fidelity)]  # first set of states
        output_states2 = output_state_all[int(self.num_fidelity):]  # second set of states

        # compute the fidelities between the two sets of states
        fidelities = self.compute_fidelity(output_states1, output_states2)

        # map the computed fidelities to the corresponding bins
        bin_index = np.floor(fidelities * self.num_bins).astype(int)  # Map fidelities to bins
        num = []
        # count the number of fidelities in each bin
        for i in range(0, self.num_bins):
            num.append(len(bin_index[bin_index == i]))
        # compute the probability distribution over the bins
        p_fidelity = np.array(num) / sum(num)

        # compute the Kullback-Leibler (K-L) divergence between the computed p_fidelity and the Haar measure
        express = stats.entropy(p_fidelity, self.p_fidelity_Haar)

        return express

    # evaluate state
    def circuit_eval_state(self, weight):
        qc = tc.Circuit(self.num_qubits)
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

        return qc.state()
