import argparse
import numpy as np
import os
import torch
from matplotlib import pyplot as plt


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--target_model_path', type=str, default='experiments/srresnet/target_model.pth', help='path of target model')
    parser.add_argument(
        '--blurry_neuron_txt',
        type=str,
        default='results/Interpret/neuron-search/srresnet/Set14/faig/blur_index.txt',
        help='file path of blurry_neuron_txt')
    parser.add_argument(
        '--noisy_neuron_txt',
        type=str,
        default='results/Interpret/neuron-search/srresnet/Set14/faig/noise_index.txt',
        help='file path of noisy_neuron_txt')
    args = parser.parse_args()

    # configuration
    target_model_path = args.target_model_path
    blurry_neuron_txt = args.blurry_neuron_txt
    noisy_neuron_txt = args.noisy_neuron_txt

    blurry_neurons = np.loadtxt(blurry_neuron_txt, dtype=int)
    noisy_neurons = np.loadtxt(noisy_neuron_txt, dtype=int)

    # define target_state_dict
    target_state_dict = torch.load(target_model_path)['params_ema']

    cumulate_num_neurons = [0]
    conv_name_list = []

    # Note that we exclude bias
    for key, value in target_state_dict.items():
        if key.find('weight') != -1:
            conv_name_list.append(key)
            num_neurons = value.size(0) * value.size(1)
            cumulate_num_neurons.append(cumulate_num_neurons[-1] + num_neurons)

    # del the first element in cumulate_num_neurons
    del cumulate_num_neurons[0]

    # ==================== deal with noisy neurons ==================== #
    select_noisy_neurons = noisy_neurons[:1519]
    # locate these neurons
    # calculate the location of these neurons
    noisy_layer_neuron = []
    noisy_row_neuron = []
    noisy_column_neuron = []
    for neuron_index in select_noisy_neurons:
        if neuron_index < 192:
            layer = 0
            key = conv_name_list[layer]
            row = neuron_index // 3
            column = neuron_index % 3
            noisy_layer_neuron.append(layer)
            noisy_row_neuron.append(row)
            noisy_column_neuron.append(column)
        else:
            for i in range(1, len(cumulate_num_neurons)):
                if neuron_index < cumulate_num_neurons[i]:
                    layer = i
                    key = conv_name_list[layer]
                    row = (neuron_index - cumulate_num_neurons[i - 1]) // target_state_dict[key].size(1)
                    column = (neuron_index - cumulate_num_neurons[i - 1]) % target_state_dict[key].size(1)
                    noisy_layer_neuron.append(layer)
                    noisy_row_neuron.append(row)
                    noisy_column_neuron.append(column)
                    break

    for neuron_idx in range(len(noisy_layer_neuron)):
        layer = noisy_layer_neuron[neuron_idx]
        row = noisy_row_neuron[neuron_idx]
        column = noisy_column_neuron[neuron_idx]

    # define the column in each layer
    each_layer_columns = []
    for conv_name in conv_name_list:
        column = target_state_dict[conv_name].size(1)
        each_layer_columns.append(column)

    noisy_map = np.zeros((256 * 64, 35))

    repeat_num_l = []
    for _, conv_name in enumerate(conv_name_list):
        weight = target_state_dict[conv_name]
        value = torch.sum(torch.sum(abs(weight), dim=2), dim=2).numpy()
        value = value.flatten()
        repeat_num = int((256 * 64) // value.shape[0])
        value = np.repeat(value, repeat_num)
        repeat_num_l.append(repeat_num)

    for neuron_idx in range(len(noisy_layer_neuron)):
        layer = noisy_layer_neuron[neuron_idx]
        row = noisy_row_neuron[neuron_idx]
        column = noisy_column_neuron[neuron_idx]
        origin_index = row * each_layer_columns[layer] + column
        if layer > 34:
            continue
        else:
            repeat_num = repeat_num_l[layer]
            noisy_map[origin_index * repeat_num:origin_index * repeat_num + min(4, repeat_num), layer] += 500
    os.makedirs('results/Interpret/curve', exist_ok=True)
    fig, ax0 = plt.subplots(1, 1)
    ax0.pcolor(noisy_map)
    fig.set_size_inches(37, 21, forward=True)
    fig.tight_layout()
    plt.axis('off')
    plt.savefig('results/Interpret/curve/noise_distribution.png')
    plt.show()

    # ==================== deal with blurry neurons ==================== #
    select_blurry_neurons = blurry_neurons[:1519]
    # locate these neurons
    # calculate the location of these neurons
    blurry_layer_neuron = []
    blurry_row_neuron = []
    blurry_column_neuron = []
    for neuron_index in select_blurry_neurons:
        if neuron_index < 192:
            layer = 0
            key = conv_name_list[layer]
            row = neuron_index // 3
            column = neuron_index % 3
            blurry_layer_neuron.append(layer)
            blurry_row_neuron.append(row)
            blurry_column_neuron.append(column)
        else:
            for i in range(1, len(cumulate_num_neurons)):
                if neuron_index < cumulate_num_neurons[i]:
                    layer = i
                    key = conv_name_list[layer]
                    row = (neuron_index - cumulate_num_neurons[i - 1]) // target_state_dict[key].size(1)
                    column = (neuron_index - cumulate_num_neurons[i - 1]) % target_state_dict[key].size(1)
                    blurry_layer_neuron.append(layer)
                    blurry_row_neuron.append(row)
                    blurry_column_neuron.append(column)
                    break

    for neuron_idx in range(len(blurry_layer_neuron)):
        layer = blurry_layer_neuron[neuron_idx]
        row = blurry_row_neuron[neuron_idx]
        column = blurry_column_neuron[neuron_idx]

    # define the column in each layer
    each_layer_columns = []
    for conv_name in conv_name_list:
        column = target_state_dict[conv_name].size(1)
        each_layer_columns.append(column)

    blurry_map = np.zeros((256 * 64, 35))

    repeat_num_l = []
    for _, conv_name in enumerate(conv_name_list):
        weight = target_state_dict[conv_name]
        value = torch.sum(torch.sum(abs(weight), dim=2), dim=2).numpy()
        value = value.flatten()
        repeat_num = int((256 * 64) // value.shape[0])
        value = np.repeat(value, repeat_num)
        repeat_num_l.append(repeat_num)

    for neuron_idx in range(len(blurry_layer_neuron)):
        layer = blurry_layer_neuron[neuron_idx]
        row = blurry_row_neuron[neuron_idx]
        column = blurry_column_neuron[neuron_idx]
        origin_index = row * each_layer_columns[layer] + column
        if layer > 34:
            continue
        else:
            repeat_num = repeat_num_l[layer]
            blurry_map[origin_index * repeat_num:origin_index * repeat_num + min(4, repeat_num), layer] += 500
    fig, ax0 = plt.subplots(1, 1)
    ax0.pcolor(blurry_map)
    fig.set_size_inches(37, 21, forward=True)
    fig.tight_layout()
    plt.axis('off')
    plt.savefig('results/Interpret/curve/blur_distribution.png')
    plt.show()


if __name__ == '__main__':
    main()
