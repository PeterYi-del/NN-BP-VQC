import os

import numpy as np
import matplotlib.pyplot as plt
import torch
from Training import compute_cost_function as cf


def plot_loss_landscape(model, num_qubits, model_name, n):

    # get all parameters of the model and flattened
    params = [param.data.numpy().flatten() for param in model.parameters()]
    flattened_params = np.concatenate(params)

    # compute the L2 norm of the flattened parameter vector
    param_norm = np.linalg.norm(flattened_params)

    # generate two random direction vectors and normalize them
    # generate two random directions of the same shape as the flattened parameter vector
    direction1 = np.random.randn(flattened_params.size)
    direction2 = np.random.randn(flattened_params.size)

    # normalized
    direction1 = (direction1 / np.linalg.norm(direction1)) * param_norm
    direction2 = (direction2 / np.linalg.norm(direction2)) * param_norm

    # set up the grid and initializing the loss landscape
    epsilon, num_points = 0.5, 200
    x_axis = np.linspace(-epsilon, epsilon, num_points)  # x axis
    y_axis = np.linspace(-epsilon, epsilon, num_points)  # y axis
    loss_landscape = np.zeros((num_points, num_points))  # initial loss landscape grid

    # compute the loss value for each grid point
    for i, x in enumerate(x_axis):
        for j, y in enumerate(y_axis):
            # compute each new parameter
            new_params = flattened_params + x * direction1 + y * direction2

            start = 0
            # bring the new parameters back into the model
            with torch.no_grad():
                for param in model.parameters():
                    size = param.numel()  # get the shape of the current parameters of the model
                    # constructed into the same shape as the current parameters,
                    # and the new parameters overwrite the old parameters
                    param.copy_(torch.tensor(new_params[start:start + size].reshape(param.shape), dtype=torch.float32))
                    start += size
            output = model(0, 'loss')
            loss = cf.cost_function(output, num_qubits)
            loss_landscape[i, j] = loss.item()

    # plot the loss landsacpe
    X, Y = np.meshgrid(x_axis, y_axis)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, loss_landscape, cmap='viridis')
    plt.xlabel('direction 1')
    plt.ylabel('direction 2')
    ax.set_zlabel('loss')

    # save figures
    output_dir = 'result/Loss_Landscape'
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f'{output_dir}/{model_name}_{num_qubits}qubits_{n + 1}.pdf')
    plt.close()

    del param, direction1, direction2, x_axis, y_axis, loss_landscape
