# noise_parameters.py

import numpy as np
import tensorcircuit as tc

# probability of noise for single-qubit, two-qubit, and bit-flip errors
single_qubit_p = 0.001
two_qubit_p = 0.01
bit_flip_p = 0.001

# create a single-qubit depolarizing channel with a probability of noise `p` divided by 3
single_qubit_dep_channel = tc.channels.generaldepolarizingchannel(p=single_qubit_p/3, num_qubits=1)

# create a two-qubit depolarizing channel with a probability of noise `p` divided by 15
two_qubit_dep_channel = tc.channels.generaldepolarizingchannel(p=two_qubit_p/15, num_qubits=2)

# kraus-operators for a bit-flip error with probability `bit_flip_p`
# K0: identity operator scaled by the square root of the non-error probability (1 – bit_flip_p)
K0 = np.array([[1, 0], [0, 1]]) * np.sqrt(1 - bit_flip_p)

# K1: bit-flip operator (Pauli-X) scaled by the square root of the error probability (bit_flip_p)
K1 = np.array([[0, 1], [1, 0]]) * np.sqrt(bit_flip_p)
