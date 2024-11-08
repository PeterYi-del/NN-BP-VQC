from qiskit.quantum_info import random_statevector


def create_random_state(num_qubits):
    # get a random input quantum state
    input_state = random_statevector(2 ** num_qubits)
    return input_state
