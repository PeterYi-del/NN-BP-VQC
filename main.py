import sys
from Training import traing_part
import config

num_qubits = 6  # the number of qubits
n = 10  # total training loops

nMax = config.problem['problems']['nMax']  # maximum number of epochs (iterations)
model_list = config.problem['problems']['model_list']  # list of models
num_bins = config.problem['problems']['num_bins']  # number of bins for expressibility calculation
num_fidelity = config.problem['problems']['num_fidelity']  # number of fidelities for expressibility calculation
circuit_state = 'pure'  # the circuit state is set to 'pure'


print("Qubits: " + str(num_qubits) + " - " + "Active Function: PReLu" + " - " + "Max Epoch: " + str(nMax) + " - " + "Model Name: " + str(model_list) + " - " + "Num Bins: " + str(num_bins) + " - " + "Num Fidelity: " + str(num_fidelity) + " - " + "Circuit state: " + str(circuit_state))
print("Begin Training")

# call the main training function, passing all the relevant parameters
traing_part.main_training_part(num_qubits, nMax, model_list, num_bins, num_fidelity, n, circuit_state)

print("End Training")

