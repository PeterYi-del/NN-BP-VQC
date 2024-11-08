import math
import os

import numpy as np
from matplotlib import pyplot as plt, ticker
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from Drawing import compute_average_loss


def plot_avg_iteration_vs_loss(loss_value, num_epoch, num_qubits, model_list):

    # compute the list of average, max, min
    avg_list, max_list, min_list, max_len = compute_average_loss.compute_average_loss(loss_value, num_epoch, model_list)

    # set the style for each model
    model_properties = {
        'NEQC-NN': ('#e87072', '-', '#f9b8bc'),
        'NEQC-CNN': ('#f7d323', '-', '#ffd19d'),
        'SQC': ('#2d4f74', '--', '#d5e9f4'),
    }

    # polt average line for each model
    for model_idx, model in enumerate(model_list):
        color, linestyle, fill_color = model_properties.get(model, (None, None, None))
        if color is None:
            print(f"Warning: Model {model} not recognized.")
            continue

        # plot average line
        plt.plot(max_len[model_idx], avg_list[model_idx], label=model, color=color, linestyle=linestyle, linewidth=2.5)
        # plot the shadow between min and max
        plt.fill_between(max_len[model_idx], min_list[model_idx], max_list[model_idx], color=fill_color, alpha=0.4)

    plt.grid(axis='y', linestyle='--', color='grey', linewidth=0.5)

    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-3, 4))
    plt.gca().yaxis.set_major_formatter(formatter)
    plt.legend(loc='upper right')

    plt.xlabel("iteration")
    plt.ylabel("average loss")

    # create a small graph to observe the convergence part
    inset_ax = inset_axes(plt.gca(), width="40%", height="30%", loc='center right', borderpad=0.5)
    for model_idx, model in enumerate(model_list):
        if model not in ['NEQC-NN', 'NEQC-CNN']:
            continue

        color, linestyle, fill_color = model_properties.get(model, (None, None, None))
        if color is None:
            continue

        # plot average loss less than 0.05
        threshold_idx = np.argmax(np.array(avg_list[model_idx]) < 0.05)

        inset_ax.plot(max_len[model_idx][threshold_idx:], avg_list[model_idx][threshold_idx:], color=color,
                      linestyle=linestyle,
                      linewidth=1.5)

    output_dir = 'result/Avg_loss_vs_epcoh'
    file_path = os.path.join(output_dir, f'Avg_loss_vs_epoch_{num_qubits}qubits.pdf')

    os.makedirs(output_dir, exist_ok=True)  # create file
    plt.savefig(file_path)

    plt.close()
