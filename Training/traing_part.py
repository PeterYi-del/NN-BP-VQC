import json
import os
import time

import numpy as np
from torch import optim
from tqdm import tqdm

from Preparing import create_random_circuit as crc, create_random_state as crs, compute_circuit_blocks as ccl, \
    model as cnn
from Training import compute_cost_function as ccf, compute_express, record_model_parameters
from Drawing import plot_avg_iteration_vs_loss, polt_loss_landscape as dll


def main_training_part(num_qubits, nMax, model_list, num_bins, num_fidelity, n, circuit_state):
    epoch_max = nMax  # maximum number of epochs that can be reached

    C = 0.001  # C value

    converge_difference = 0.0001  # the threshold for determining convergence

    check_converge_per_iter = 100  # check convergence every 100 iterations

    loss_last = 1e10  # initialize loss_last with a large value to calculate the difference between the current and
    # last loss

    # get the number of circuit blocks based on the number of qubits
    num_block = ccl.compute_circuit_block(num_qubits)

    # get the number of models
    num_models = len(model_list)

    # initialize lists to store loss values for both NEQC and SQC across different models
    loss_value = [[[] for _ in range(n)] for _ in range(num_models)]  # loss values for NEQC and SQC

    # initialize lists to store epoch values for both NEQC and SQC across different models
    num_epoch = [[[] for _ in range(n)] for _ in range(num_models)]  # epoch values for NEQC and SQC

    # lists to store the expressibility for NEQC and SQC, respectively
    NEQC_express = []  # record the expressibility of NEQC for each qubit
    SQC_express = []  # record the expressibility of SQC for each qubit

    # tenfold training loop
    start_time = time.time()  # start the timer to record training time
    for i in range(n):
        # generate a random circuit and input state for the current qubit and block configuration
        circuit = crc.create_random_circuit(num_qubits, num_block)  # generate a random quantum circuit
        input_state = crs.create_random_state(num_qubits)  # generate a random quantum state

        for model_idx, model_name in enumerate(model_list):

            # initialize the model based on its name
            if model_name == 'SQC':
                model = cnn.SQC(num_qubits, num_block, circuit, input_state, circuit_state)  # initialize SQC model
            elif model_name == 'NEQC-NN':
                model = cnn.NEQC_NN(num_qubits, num_block, circuit, input_state,
                                    circuit_state)  # initialize NEQC-NN model

            elif model_name == 'NEQC-CNN':
                model = cnn.NEQC_CNN(num_qubits, num_block, circuit, input_state,
                                     circuit_state)  # initialize NEQC-CNN model
                record_model_parameters.record_model_parameters(model, num_qubits, model_name)
            # select the optimizer for training
            optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)  # Stochastic Gradient Descent optimizer

            # start training
            with tqdm(desc="Training", dynamic_ncols=True) as pbar:  # progress bar for training
                for nepoch in range(epoch_max):
                    optimizer.zero_grad()  # zero the gradients to prepare for the new iteration
                    probs = model(0,
                                  'training')  # run the model in training mode, no alpha input needed (0 is
                    # placeholder)
                    cost = ccf.cost_function(probs, num_qubits)  # compute the cost function
                    cost.backward()  # backward to compute gradients
                    optimizer.step()  # perform a step of optimization
                    ex = cost.item()  # record the current cost value (C)

                    # update the progress bar description with the current training status
                    pbar.set_description(
                        f"Qubits: {num_qubits}, Epoch: {nepoch}, Cost: {ex:.4f}, Model: {model_name}, N:{i + 1}")

                    # record loss values every 10 epochs
                    if nepoch % 10 == 0:
                        num_epoch[model_idx][i].append(nepoch)  # store the current epoch value
                        loss_value[model_idx][i].append(ex)  # store the current loss value

                    # check if the cost has reached the target threshold (C)
                    if ex < C:
                        break

                    # check for convergence every 'check_converge_per_iter' iterations
                    if nepoch % check_converge_per_iter == 0:
                        distance = abs(loss_last - ex)  # compute the change in loss from the last iteration
                        if distance < converge_difference:  # if the change is smaller than the threshold, stop training
                            break
                        else:
                            loss_last = ex  # update the last loss value

                    # check if the maximum number of epochs has been exceeded
                    if nepoch > epoch_max:
                        break
                    else:
                        nepoch += 1  # increase the epoch counter

            # after training, save the model's final state and additional data
            output_dir = 'result/express'  # directory to save the results
            os.makedirs(output_dir, exist_ok=True)  # create the directory if it doesn't already exist

            # for NEQC models, compute and save expressibility and losslandscape
            if model_name == 'NEQC-NN' or model_name == 'NEQC-CNN':
                neqc_express = compute_express.computeNeqcExpress(num_bins, num_qubits, num_fidelity, circuit,
                                                                  circuit_state)  # compute expressibility for NEQC
                NEQC_express.append(
                    neqc_express.compute_neqc_express(model, model_name))  # store the expressibility in NEQC_express
                dll.plot_loss_landscape(model, num_qubits, model_name, i)  # draw the losslandscape for NEQC
                # save the expressibility data to a file
                np.savetxt('result/express/{}_express_list_{}qubits_{}.txt'.format(model_name, num_qubits, i),
                           NEQC_express)

            # for SQC model, compute and save expressibility and losslandscape
            elif model_name == 'SQC':
                sqc_express = compute_express.computeSqcExpress(num_bins, num_qubits, num_fidelity, circuit,
                                                                circuit_state)  # compute expressibility for SQC
                SQC_express.append(
                    sqc_express.compute_sqc_express(num_block))  # store the expressibility in SQC_express
                dll.plot_loss_landscape(model, num_qubits, model_name, i)  # draw the loss landscape for SQC
                # save the expressiveness data to a file
                np.savetxt('result/express/{}_express_list_{}qubits_{}.txt'.format(model_name, num_qubits, i),
                           SQC_express)

            # delete the model to free memory after training is complete
            del model

    # end the timer after the training process is complete
    end_time = time.time()

    # record the training time for the tenfold cross-validation
    np.savetxt('result/time_{}.txt'.format(num_qubits), [end_time - start_time])
    # record the loss value and itertaion
    with open('result/loss_{}.json'.format(num_qubits), 'w') as f:
        json.dump(loss_value, f)
    with open('result/iteration_{}.json'.format(num_qubits), 'w') as f:
        json.dump(num_epoch, f)
    # plot the average iteration vs. loss graph
    plot_avg_iteration_vs_loss.plot_avg_iteration_vs_loss(loss_value, num_epoch, num_qubits, model_list)
