from math import log, ceil


def compute_circuit_block(num_qubits):
    # take (n^2log(n)) and round up
    num_block = ceil(num_qubits ** 2 * log(num_qubits))
    return num_block
