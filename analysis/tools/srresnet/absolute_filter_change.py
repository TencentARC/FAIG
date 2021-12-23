import argparse
import numpy as np
import os
import torch


def absolute_filter_change(baseline_state_dict, target_state_dict):
    """ Calculate sum(abs(K2 - K1) / sum(K1))

    Args:
        baseline_state_dict (dict): state_dict of ori_net
        target_state_dict (dict): state_dict of finetune_net

    Returns:
        sorted_diff (list): sorted values
        sorted_index (list): sorted index of kernel
    """

    # save all weight to list
    baseline_weight_list = []
    for key, value in baseline_state_dict.items():
        if key.find('weight') != -1:
            weight = value.reshape(-1, 3, 3)
            baseline_weight_list.append(weight)
    # [-1, 3, 3]
    baseline_weight_list = torch.cat(baseline_weight_list, dim=0)

    target_weight_list = []
    for key, value in target_state_dict.items():
        if key.find('weight') != -1:
            weight = value.reshape(-1, 3, 3)
            target_weight_list.append(weight)
    # [-1, 3, 3]
    target_weight_list = torch.cat(target_weight_list, dim=0)

    sum_baseline_weight = torch.sum(torch.sum(abs(baseline_weight_list), dim=1), dim=1)
    sum_baseline_weight = sum_baseline_weight.unsqueeze(1).unsqueeze(1)

    diff = torch.sum(torch.sum(abs(target_weight_list - baseline_weight_list) / sum_baseline_weight, dim=1), dim=1)

    return diff.cpu().numpy()


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--baseline_model_path',
        type=str,
        default='experiments/srresnet/baseline_model.pth',
        help='path of baseline model')
    parser.add_argument(
        '--target_model_path', type=str, default='experiments/srresnet/target_model.pth', help='path of target model')
    parser.add_argument(
        '--record_filters_folder',
        type=str,
        default='results/Interpret/neuron-search/srresnet/Set14/abs_filter_change',
        help='folder that saves the sorted location index of discovered filters')
    args = parser.parse_args()

    # configuration
    baseline_model_path = args.baseline_model_path
    target_model_path = args.target_model_path
    record_filters_folder = args.record_filters_folder
    os.makedirs(record_filters_folder, exist_ok=True)

    baseline_state_dict = torch.load(baseline_model_path)['params_ema']
    target_state_dict = torch.load(target_model_path)['params_ema']

    # calculate the neurons
    diff = absolute_filter_change(baseline_state_dict, target_state_dict)

    sorted_location = np.argsort(diff)[::-1]
    save_noisy_filter_txt = os.path.join(record_filters_folder, 'noise_index.txt')
    np.savetxt(save_noisy_filter_txt, sorted_location, delimiter=',', fmt='%d')

    save_blurry_filter_txt = os.path.join(record_filters_folder, 'blur_index.txt')
    np.savetxt(save_blurry_filter_txt, sorted_location, delimiter=',', fmt='%d')


if __name__ == '__main__':
    main()
