import argparse
import cv2
import glob
import numpy as np
import os
import re
import torch

from archs.srcnn_style_arch import srcnn_style_net
from basicsr.utils import img2tensor, tensor2img


def mask_neurons_with_different_proportion(baseline_model_path,
                                           target_model_path,
                                           noisy_neuron_txt,
                                           blurry_neuron_txt,
                                           dataset_root,
                                           blur_folder_name,
                                           noise_folder_name,
                                           save_mask_filter_folder,
                                           scale=2):
    """ Replace the weights of discovered filters in target model
        with those filters in baseline model (at the same location)

    Args:
        baseline_model_path (str): path of baseline model
        target_model_path (str): path of target model
        noisy_neuron_txt (str): path of saved sorted filters for noise degradation
        blurry_neuron_txt (str): path of saved sorted filters for blur degradation
        dataset_root (str): root path of dataset
        blur_folder_name (str): folder name of blurry images
        noise_folder_name (str): folder name of noisy images
        save_mask_filter_folder (str): folder path of restoration results of masked model
        scale (int)

    """

    # configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    blur_folder = os.path.join(dataset_root, blur_folder_name)
    noise_folder = os.path.join(dataset_root, noise_folder_name)

    baseline_state_dict = torch.load(baseline_model_path)['params_ema']

    target_state_dict = torch.load(target_model_path)['params_ema']
    target_net = srcnn_style_net(scale=2)
    target_net.eval()
    target_net = target_net.to(device)
    target_net.load_state_dict(target_state_dict)

    conv_index = [0, 2, 4, 6, 8, 10, 12, 15, 17]
    cumulate_num_neurons = [192, 8384, 24768, 41152, 57536, 73920, 139456, 155840, 156224]

    # load blurry and noisy neurons
    blurry_neurons = np.loadtxt(blurry_neuron_txt, dtype=int)
    noisy_neurons = np.loadtxt(noisy_neuron_txt, dtype=int)
    total_neuron_nums = len(blurry_neurons)

    noise_img_list = sorted(glob.glob(os.path.join(noise_folder, '*')))
    blur_img_list = sorted(glob.glob(os.path.join(blur_folder, '*')))

    for proportion in [1, 3, 5, 10]:
        print(f'Processing the {proportion}% masking proportion')

        selected_num_neurons = int(total_neuron_nums * proportion / 100)

        save_neuron_folder = f'{save_mask_filter_folder}/{selected_num_neurons}kernels'
        os.makedirs(save_neuron_folder, exist_ok=True)

        # ==================== deal with noisy neurons ==================== #
        select_noisy_neurons = noisy_neurons[:selected_num_neurons]
        save_sub_folder = f'{save_neuron_folder}/maskdenoisefilter'
        os.makedirs(save_sub_folder, exist_ok=True)

        # calculate the location of these neurons
        noisy_layer_neuron = []
        noisy_row_neuron = []
        noisy_column_neuron = []

        for neuron_index in select_noisy_neurons:
            if neuron_index < 192:
                layer = 0
                row = neuron_index // 3
                column = neuron_index % 3
                noisy_layer_neuron.append(layer)
                noisy_row_neuron.append(row)
                noisy_column_neuron.append(column)
            else:
                for i in range(0, len(cumulate_num_neurons)):
                    if neuron_index < cumulate_num_neurons[i]:
                        layer = conv_index[i]
                        row = (neuron_index - cumulate_num_neurons[i - 1]) // target_net.features[layer].weight.size(1)
                        column = (neuron_index -
                                  cumulate_num_neurons[i - 1]) % target_net.features[layer].weight.size(1)
                        noisy_layer_neuron.append(layer)
                        noisy_row_neuron.append(row)
                        noisy_column_neuron.append(column)
                        break

        # replace param in new_net_state_dict with ori_net_state_dict
        new_net_state_dict = {}

        # locate the params
        for key, _ in target_state_dict.items():
            new_net_state_dict[key] = target_state_dict[key].clone()

            # locate layer
            current_layer_idx = int(re.findall(r'\d+', key)[0])
            if current_layer_idx in noisy_layer_neuron:
                # locate weight params
                if key.find('weight') != -1:
                    for i in range(len(noisy_layer_neuron)):
                        if noisy_layer_neuron[i] == current_layer_idx:
                            new_net_state_dict[key][noisy_row_neuron[i],
                                                    noisy_column_neuron[i], :, :] = baseline_state_dict[key][
                                                        noisy_row_neuron[i], noisy_column_neuron[i], :, :]

        net = srcnn_style_net(scale=scale)
        net.eval()
        net = net.to(device)
        net.load_state_dict(new_net_state_dict, strict=True)

        for noisy_img_path in noise_img_list:
            # read noisy image
            imgname = os.path.splitext(os.path.basename(noisy_img_path))[0]
            noisy_img = cv2.imread(noisy_img_path, cv2.IMREAD_COLOR).astype(np.float32) / 255.
            noisy_img = img2tensor(noisy_img).unsqueeze(0).to(device)

            # inference
            with torch.no_grad():
                noisy_output = net(noisy_img).squeeze(0)
            # save image
            noisy_output = tensor2img(noisy_output, min_max=(0, 1))

            output_save_folder = f'{save_sub_folder}/{noise_folder_name}'
            os.makedirs(output_save_folder, exist_ok=True)
            cv2.imwrite(f'{output_save_folder}/{imgname}.png', noisy_output)

            # read blur image
            blurry_img_path = f'{blur_folder}/{imgname}.png'
            blurry_img = cv2.imread(blurry_img_path, cv2.IMREAD_COLOR).astype(np.float32) / 255.
            blurry_img = img2tensor(blurry_img).unsqueeze(0).to(device)
            # inference
            with torch.no_grad():
                blurry_output = net(blurry_img).squeeze(0)
            # save image
            blurry_output = tensor2img(blurry_output, min_max=(0, 1))
            output2_save_folder = f'{save_sub_folder}/{blur_folder_name}'
            os.makedirs(output2_save_folder, exist_ok=True)
            cv2.imwrite(f'{output2_save_folder}/{imgname}.png', blurry_output)

        # # ==================== deal with blurry neurons ==================== #
        select_blurry_neurons = blurry_neurons[:selected_num_neurons]
        save_sub_folder = f'{save_neuron_folder}/maskdeblurfilter'
        os.makedirs(save_sub_folder, exist_ok=True)

        # calculate the location of these neurons
        blurry_layer_neuron = []
        blurry_row_neuron = []
        blurry_column_neuron = []
        for neuron_index in select_blurry_neurons:
            if neuron_index < 192:
                layer = 0
                row = neuron_index // 3
                column = neuron_index % 3
                blurry_layer_neuron.append(layer)
                blurry_row_neuron.append(row)
                blurry_column_neuron.append(column)
            else:
                for i in range(0, len(cumulate_num_neurons)):
                    if neuron_index < cumulate_num_neurons[i]:
                        layer = conv_index[i]
                        row = (neuron_index - cumulate_num_neurons[i - 1]) // target_net.features[layer].weight.size(1)
                        column = (neuron_index -
                                  cumulate_num_neurons[i - 1]) % target_net.features[layer].weight.size(1)
                        blurry_layer_neuron.append(layer)
                        blurry_row_neuron.append(row)
                        blurry_column_neuron.append(column)
                        break

        # replace param in new_net_state_dict with ori_net_state_dict
        new_net_state_dict = {}

        # locate the params
        for key, _ in target_state_dict.items():
            new_net_state_dict[key] = target_state_dict[key].clone()

            # locate layer
            current_layer_idx = int(re.findall(r'\d+', key)[0])
            if current_layer_idx in blurry_layer_neuron:
                # locate weight params
                if key.find('weight') != -1:
                    for i in range(len(blurry_layer_neuron)):
                        if blurry_layer_neuron[i] == current_layer_idx:
                            new_net_state_dict[key][blurry_row_neuron[i],
                                                    blurry_column_neuron[i], :, :] = baseline_state_dict[key][
                                                        blurry_row_neuron[i], blurry_column_neuron[i], :, :]

        net = srcnn_style_net(scale=scale)
        net.eval()
        net = net.to(device)
        net.load_state_dict(new_net_state_dict)

        for blurry_img_path in blur_img_list:
            # read image
            imgname = os.path.splitext(os.path.basename(blurry_img_path))[0]
            blurry_img = cv2.imread(blurry_img_path, cv2.IMREAD_COLOR).astype(np.float32) / 255.
            blurry_img = torch.from_numpy(np.transpose(blurry_img[:, :, [2, 1, 0]], (2, 0, 1))).float()
            blurry_img = blurry_img.unsqueeze(0).to(device)

            # inference
            with torch.no_grad():
                blurry_output = net(blurry_img).squeeze(0)
            # save image
            blurry_output = tensor2img(blurry_output, min_max=(0, 1))

            output_save_folder = f'{save_sub_folder}/{blur_folder_name}'
            os.makedirs(output_save_folder, exist_ok=True)
            cv2.imwrite(f'{output_save_folder}/{imgname}.png', blurry_output)

            # read noisy image
            noisy_img_path = f'{noise_folder}/{imgname}.png'
            noisy_img = cv2.imread(noisy_img_path, cv2.IMREAD_COLOR).astype(np.float32) / 255.
            noisy_img = torch.from_numpy(np.transpose(noisy_img[:, :, [2, 1, 0]], (2, 0, 1))).float()
            noisy_img = noisy_img.unsqueeze(0).to(device)
            # inference
            with torch.no_grad():
                noisy_output = net(noisy_img).squeeze(0)
            # save image
            noisy_output = tensor2img(noisy_output, min_max=(0, 1))
            output2_save_folder = f'{save_sub_folder}/{noise_folder_name}'
            os.makedirs(output2_save_folder, exist_ok=True)
            cv2.imwrite(f'{output2_save_folder}/{imgname}.png', noisy_output)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--baseline_model_path',
        type=str,
        default='experiments/srcnn_style/baseline_model.pth',
        help='path of baseline model')
    parser.add_argument(
        '--target_model_path',
        type=str,
        default='experiments/srcnn_style/target_model.pth',
        help='path of target model')
    parser.add_argument('--scale', type=int, default=2, help='scale ratio')
    parser.add_argument('--dataset', type=str, default='Set14', help='dataset name')
    parser.add_argument('--blur_folder_name', type=str, default='Blur2_LRbicx2', help='name of blurry folder')
    parser.add_argument('--noise_folder_name', type=str, default='LRbicx2_noise0.1', help='name of noisy folder')
    parser.add_argument(
        '--srcnn_neuron_folder',
        type=str,
        default='results/Interpret/neuron-search/srcnn_style',
        help='folder that contains the discvoered important neurons of different methods in SRResNet')
    parser.add_argument(
        '--srcnn_mask_save_folder',
        type=str,
        default='results/Interpret/masking/srcnn_style',
        help='folder that saves the masking results')
    args = parser.parse_args()

    baseline_model_path = args.baseline_model_path
    target_model_path = args.target_model_path

    scale = args.scale
    dataset = args.dataset
    blur_folder_name = args.blur_folder_name
    noise_folder_name = args.noise_folder_name
    srcnn_neuron_folder = args.srcnn_neuron_folder
    srcnn_mask_save_folder = args.srcnn_mask_save_folder
    dataset_root = f'datasets/{dataset}'

    # faig
    faig_noisy_neuron_txt = f'{srcnn_neuron_folder}/{dataset}/faig/noise_index.txt'
    faig_blurry_neuron_txt = f'{srcnn_neuron_folder}/{dataset}/faig/blur_index.txt'
    save_mask_faig_filters_folder = f'{srcnn_mask_save_folder}/{dataset}/faig'
    os.makedirs(save_mask_faig_filters_folder, exist_ok=True)

    # ig
    ig_noisy_neuron_txt = f'{srcnn_neuron_folder}/{dataset}/ig/noise_index.txt'
    ig_blurry_neuron_txt = f'{srcnn_neuron_folder}/{dataset}/ig/blur_index.txt'
    save_mask_ig_filters_folder = f'{srcnn_mask_save_folder}/{dataset}/ig'
    os.makedirs(save_mask_ig_filters_folder, exist_ok=True)

    # abs_filter_change
    abs_filter_change_noisy_neuron_txt = f'{srcnn_neuron_folder}/{dataset}/abs_filter_change/noise_index.txt'
    abs_filter_change_blurry_neuron_txt = f'{srcnn_neuron_folder}/{dataset}/abs_filter_change/blur_index.txt'
    save_mask_abs_filter_change_filters_folder = f'{srcnn_mask_save_folder}/{dataset}/abs_filter_change'
    os.makedirs(save_mask_abs_filter_change_filters_folder, exist_ok=True)

    # random
    random_noisy_neuron_txt = f'{srcnn_neuron_folder}/{dataset}/random/noise_index.txt'
    random_blurry_neuron_txt = f'{srcnn_neuron_folder}/{dataset}/random/blur_index.txt'
    save_mask_random_filters_folder = f'{srcnn_mask_save_folder}/{dataset}/random'
    os.makedirs(save_mask_random_filters_folder, exist_ok=True)

    print('Now we are masking faig-discovered filters.')
    # replace the neurons of faig methods
    mask_neurons_with_different_proportion(
        baseline_model_path=baseline_model_path,
        target_model_path=target_model_path,
        noisy_neuron_txt=faig_noisy_neuron_txt,
        blurry_neuron_txt=faig_blurry_neuron_txt,
        dataset_root=dataset_root,
        blur_folder_name=blur_folder_name,
        noise_folder_name=noise_folder_name,
        save_mask_filter_folder=save_mask_faig_filters_folder,
        scale=scale)

    print('Now we are masking ig-discovered filters.')
    # replace the neurons of ig methods
    mask_neurons_with_different_proportion(
        baseline_model_path=baseline_model_path,
        target_model_path=target_model_path,
        noisy_neuron_txt=ig_noisy_neuron_txt,
        blurry_neuron_txt=ig_blurry_neuron_txt,
        dataset_root=dataset_root,
        blur_folder_name=blur_folder_name,
        noise_folder_name=noise_folder_name,
        save_mask_filter_folder=save_mask_ig_filters_folder,
        scale=scale)

    print('Now we are masking abs_filter_change-discovered filters.')
    # replace the neurons of abs_filter_change methods
    mask_neurons_with_different_proportion(
        baseline_model_path=baseline_model_path,
        target_model_path=target_model_path,
        noisy_neuron_txt=abs_filter_change_noisy_neuron_txt,
        blurry_neuron_txt=abs_filter_change_blurry_neuron_txt,
        dataset_root=dataset_root,
        blur_folder_name=blur_folder_name,
        noise_folder_name=noise_folder_name,
        save_mask_filter_folder=save_mask_abs_filter_change_filters_folder,
        scale=scale)

    print('Now we are masking random-discovered filters.')
    # replace the neurons of random methods
    mask_neurons_with_different_proportion(
        baseline_model_path=baseline_model_path,
        target_model_path=target_model_path,
        noisy_neuron_txt=random_noisy_neuron_txt,
        blurry_neuron_txt=random_blurry_neuron_txt,
        dataset_root=dataset_root,
        blur_folder_name=blur_folder_name,
        noise_folder_name=noise_folder_name,
        save_mask_filter_folder=save_mask_random_filters_folder,
        scale=scale)
