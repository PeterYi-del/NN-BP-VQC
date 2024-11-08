import numpy as np


def compute_average_loss(loss_value, num_epoch, model_list):
    num_model = len(model_list)  # number of models
    avg_list, max_list, min_list, max_len = [], [], [], []

    for model_idx in range(num_model):
        # find the max length of the loss list
        max_length = max(len(subitem) for subitem in loss_value[model_idx])

        # pad the loss list in each experiment to the same length,
        # the loss list of the experiment will be filled with its last value instead of 0
        loss_or_norm = [np.pad(trial, (0, max_length - len(trial)), 'edge') for trial in loss_value[model_idx]]
        loss_or_norm = np.array(loss_or_norm)

        avg_list.append(np.mean(loss_or_norm, axis=0).tolist())  # average
        max_list.append(np.max(loss_or_norm, axis=0).tolist())  # max
        min_list.append(np.min(loss_or_norm, axis=0).tolist())  # min

        # find the longest epoch list
        longest_epoch = max(num_epoch[model_idx], key=len)
        max_len.append(longest_epoch)

    return avg_list, max_list, min_list, max_len
