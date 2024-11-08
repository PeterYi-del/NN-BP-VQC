import os


def record_model_parameters(model, num_qubits, model_name):
    os.makedirs('result/model_params', exist_ok=True)
    if model_name == 'NEQC-NN':
        # record full connected neural network params
        params = {
            'l1_weight': model.l1.weight.data.clone(),
            'l1_bias': model.l1.bias.data.clone(),
            'l2_weight': model.l2.weight.data.clone(),
            'l2_bias': model.l2.bias.data.clone(),
            'l3_weight': model.l3.weight.data.clone(),
            'l3_bias': model.l3.bias.data.clone(),
            'layernorm1_weight': model.layernorm1.weight.data.clone(),
            'layernorm1_bias': model.layernorm1.bias.data.clone(),
            'layernorm2_weight': model.layernorm2.weight.data.clone(),
            'layernorm2_bias': model.layernorm2.bias.data.clone(),
            'layernorm3_weight': model.layernorm3.weight.data.clone(),
            'layernorm3_bias': model.layernorm3.bias.data.clone(),
        }
    elif model_name == 'NEQC-CNN':
        # record conv layer params
        params = {
            'conv1_weight': model.conv1.weight.data.clone(),
            'conv1_bias': model.conv1.bias.data.clone(),
            'conv2_weight': model.conv2.weight.data.clone(),
            'conv2_bias': model.conv2.bias.data.clone(),
            'conv3_weight': model.conv3.weight.data.clone(),
            'conv3_bias': model.conv3.bias.data.clone(),
        }

    # write into a file
    with open(f'result/model_params/{model_name}_params_{num_qubits}qubits.txt', 'w') as f:
        f.write("Conv Layer's params:\n\n")
        for name, param in params.items():
            # 写入参数名称和对应的值
            f.write(f"{name}:\n{param.numpy()}\n\n")
