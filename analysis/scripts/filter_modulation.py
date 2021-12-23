import argparse
import cv2
import glob
import numpy as np
import os
import torch
from tqdm import tqdm

from basicsr.archs.srresnet_arch import MSRResNet
from basicsr.utils import tensor2img
from basicsr.utils.img_util import img2tensor


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--baseline_model_path',
        type=str,
        default='experiments/srresnet/baseline_model.pth',
        help='path of baseline model')
    parser.add_argument(
        '--target_model_path', type=str, default='experiments/srresnet/target_model.pth', help='path of target model')
    parser.add_argument('--dataset', type=str, default='Set14', help='dataset name')
    parser.add_argument(
        '--blurry_neuron_txt',
        type=str,
        default='results/Interpret/neuron-search/srresnet/Set14/faig/blur_index.txt',
        help='file path of blurry_neuron_txt')
    parser.add_argument('--scale', type=int, default=2, help='scale ratio')
    parser.add_argument(
        '--blur_folder',
        type=str,
        default='datasets/PIRM_Self-Val_set/Blur2_LRbicx2',
        help='folder that contains blurry image')
    parser.add_argument(
        '--save_folder',
        type=str,
        default='results/Interpret/modulation/srresnet/PIRM_Self-Val_set/faig',
        help='folder that saves the modulated results')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # configuration
    baseline_model_path = args.baseline_model_path
    target_model_path = args.target_model_path
    blurry_neuron_txt = args.blurry_neuron_txt
    scale = args.scale
    blur_folder = args.blur_folder
    save_folder = args.save_folder
    os.makedirs(save_folder, exist_ok=True)

    # define ori_net_state_dict
    baseline_state_dict = torch.load(baseline_model_path)['params_ema']

    # define fintune_net_state_dict
    target_state_dict = torch.load(target_model_path)['params_ema']

    cumulate_num_neurons = [0]
    conv_name_list = []
    bias_name_list = []

    # Note that we exclude bias
    for key, value in target_state_dict.items():
        if key.find('weight') != -1:
            conv_name_list.append(key)
            num_neurons = value.size(0) * value.size(1)
            cumulate_num_neurons.append(cumulate_num_neurons[-1] + num_neurons)
        else:
            bias_name_list.append(key)

    # del the first element in cumulate_num_neurons
    del cumulate_num_neurons[0]

    # load blurry and noisy neurons
    blurry_neurons = np.loadtxt(blurry_neuron_txt, dtype=int)
    # noisy_neurons = np.loadtxt(noisy_neuron_txt, dtype=int)
    total_neuron_nums = len(blurry_neurons)

    # noise_img_list = sorted(glob.glob(os.path.join(noise_folder, '*')))
    blur_img_list = sorted(glob.glob(os.path.join(blur_folder, '*')))

    proportion = 1

    selected_num_neurons = int(total_neuron_nums * proportion / 100)

    save_neuron_folder = f'{save_folder}/{selected_num_neurons}kernels'  #
    os.makedirs(save_neuron_folder, exist_ok=True)

    # deal with noisy neurons
    select_noisy_neurons = blurry_neurons[:selected_num_neurons]
    # save_noisy_sub_folder = f'{save_neuron_folder}/Modulation-denoise-filter'
    # os.makedirs(save_noisy_sub_folder, exist_ok=True)

    save_blurry_sub_folder = f'{save_neuron_folder}/Modulation-deblur-filter'
    os.makedirs(save_blurry_sub_folder, exist_ok=True)

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

    # replace param in new_net_state_dict with ori_net_state_dict
    new_net_state_dict = {}
    for key, _ in target_state_dict.items():
        new_net_state_dict[key] = target_state_dict[key].clone()

    for coefficient in np.linspace(0, 1.5, 16):
        coefficient = round(coefficient, 2)

        # locate the params
        for i in range(len(noisy_layer_neuron)):
            # locate weight params
            layer = noisy_layer_neuron[i]
            conv_key = conv_name_list[layer]
            row = noisy_row_neuron[i]
            column = noisy_column_neuron[i]
            new_net_state_dict[conv_key][row, column, :, :] = coefficient * target_state_dict[conv_key][
                row, column, :, :] + (1 - coefficient) * baseline_state_dict[conv_key][row, column, :, :]

        net = MSRResNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=16, upscale=scale)
        net.eval()
        net = net.to(device)
        net.load_state_dict(new_net_state_dict, strict=True)

        print(f'Now the coefficient is {coefficient}')
        pbar = tqdm(total=len(blur_img_list), desc='')
        for path in blur_img_list:
            # read image
            imgname = os.path.splitext(os.path.basename(path))[0]
            noisy_img = cv2.imread(path, cv2.IMREAD_COLOR).astype(np.float32) / 255.
            noisy_img = img2tensor(noisy_img).unsqueeze(0).to(device)
            pbar.update(1)
            pbar.set_description(f'Read {imgname}')

            # inference
            with torch.no_grad():
                noisy_output = net(noisy_img).squeeze(0)
            # save image
            noisy_output = tensor2img(noisy_output, min_max=(0, 1))
            cv2.imwrite(f'{save_blurry_sub_folder}/{imgname}_{coefficient}.png', noisy_output)

        pbar.close()


if __name__ == '__main__':
    main()
