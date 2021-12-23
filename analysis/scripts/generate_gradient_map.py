import cv2
import glob
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from basicsr.utils.img_util import img2tensor


class GetGradient(nn.Module):
    """ generate the gradient map
    """

    def __init__(self):
        super(GetGradient, self).__init__()
        kernel_v = [[0, -1, 0], [0, 0, 0], [0, 1, 0]]
        kernel_h = [[0, 0, 0], [-1, 0, 1], [0, 0, 0]]
        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)
        self.weight_h = nn.Parameter(data=kernel_h, requires_grad=False)
        self.weight_v = nn.Parameter(data=kernel_v, requires_grad=False)

    def forward(self, x):
        x0 = x[:, 0]
        x1 = x[:, 1]
        x2 = x[:, 2]
        x0_v = F.conv2d(x0.unsqueeze(1), self.weight_v, padding=1)
        x0_h = F.conv2d(x0.unsqueeze(1), self.weight_h, padding=1)

        x1_v = F.conv2d(x1.unsqueeze(1), self.weight_v, padding=1)
        x1_h = F.conv2d(x1.unsqueeze(1), self.weight_h, padding=1)

        x2_v = F.conv2d(x2.unsqueeze(1), self.weight_v, padding=1)
        x2_h = F.conv2d(x2.unsqueeze(1), self.weight_h, padding=1)

        x0 = torch.sqrt(torch.pow(x0_v, 2) + torch.pow(x0_h, 2) + 1e-6)
        x1 = torch.sqrt(torch.pow(x1_v, 2) + torch.pow(x1_h, 2) + 1e-6)
        x2 = torch.sqrt(torch.pow(x2_v, 2) + torch.pow(x2_h, 2) + 1e-6)

        x = torch.cat([x0, x1, x2], dim=1)
        return x


def generate_gradient_maps_within_several_folders(net_type, target_model_output_root,
                                                  save_target_model_output_gradient_root, faig_root,
                                                  save_faig_gradient_root, ig_root, save_ig_gradient_root,
                                                  abs_filter_change_root, save_abs_filter_change_gradient_root,
                                                  random_root, save_random_gradient_root):

    # configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    grad_net = GetGradient().to(device)

    # generate the gradient map of the output of the finetuned model
    for folder in os.listdir(target_model_output_root):
        save_sub_gradient_finetune_model = os.path.join(save_target_model_output_gradient_root, folder)
        os.makedirs(save_sub_gradient_finetune_model, exist_ok=True)
        folder_path = os.path.join(target_model_output_root, folder)
        img_list = sorted(glob.glob(os.path.join(folder_path, '*.png')))
        for img_path in img_list:
            imgname = os.path.splitext(os.path.basename(img_path))[0]
            img = cv2.imread(img_path, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.
            img = np.repeat(img, 3, axis=2)
            img = img2tensor(img).unsqueeze(0).to(device)

            with torch.no_grad():
                gradient_map = grad_net(img).squeeze(0)

            gradient_map = gradient_map.detach().cpu().numpy()

            # save the gradient map
            save_img_path = f'{save_sub_gradient_finetune_model}/{imgname}.npy'
            np.save(save_img_path, gradient_map)

    if net_type == 'srcnn_style':
        selected_neuron_folders = ['1562kernels', '4686kernels', '7811kernels', '15622kernels']
    elif net_type == 'srresnet':
        selected_neuron_folders = ['1519kernels', '4558kernels', '7596kernels', '15193kernels']

    # generate the gradient map of the output of faig methods
    # for neuron_folder in os.listdir(faig_root):
    for neuron_folder in selected_neuron_folders:
        neuron_path = os.path.join(faig_root, neuron_folder)
        for sub_folder in os.listdir(neuron_path):
            sub_folder_path = os.path.join(neuron_path, sub_folder)
            for degradation_folder in os.listdir(sub_folder_path):
                degradation_path = os.path.join(sub_folder_path, degradation_folder)
                save_degradation_folder = os.path.join(save_faig_gradient_root, neuron_folder, sub_folder,
                                                       degradation_folder)
                os.makedirs(save_degradation_folder, exist_ok=True)

                img_list = sorted(glob.glob(os.path.join(degradation_path, '*.png')))
                for img_path in img_list:
                    imgname = os.path.splitext(os.path.basename(img_path))[0]
                    img = cv2.imread(img_path, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.
                    img = np.repeat(img, 3, axis=2)
                    img = img2tensor(img).unsqueeze(0).to(device)

                    with torch.no_grad():
                        gradient_map = grad_net(img).squeeze(0)

                    gradient_map = gradient_map.detach().cpu().numpy()

                    # save the gradient map
                    save_img_path = f'{save_degradation_folder}/{imgname}.npy'
                    np.save(save_img_path, gradient_map)

    # generate the gradient map of the output of ig methods
    for neuron_folder in selected_neuron_folders:
        neuron_path = os.path.join(ig_root, neuron_folder)
        for sub_folder in os.listdir(neuron_path):
            sub_folder_path = os.path.join(neuron_path, sub_folder)
            for degradation_folder in os.listdir(sub_folder_path):
                degradation_path = os.path.join(sub_folder_path, degradation_folder)
                save_degradation_folder = os.path.join(save_ig_gradient_root, neuron_folder, sub_folder,
                                                       degradation_folder)
                os.makedirs(save_degradation_folder, exist_ok=True)

                img_list = sorted(glob.glob(os.path.join(degradation_path, '*.png')))
                for img_path in img_list:
                    imgname = os.path.splitext(os.path.basename(img_path))[0]
                    img = cv2.imread(img_path, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.
                    img = np.repeat(img, 3, axis=2)
                    img = img2tensor(img).unsqueeze(0).to(device)

                    with torch.no_grad():
                        gradient_map = grad_net(img).squeeze(0)

                    gradient_map = gradient_map.detach().cpu().numpy()

                    # save the gradient map
                    save_img_path = f'{save_degradation_folder}/{imgname}.npy'
                    np.save(save_img_path, gradient_map)

    # generate the gradient map of the output of abs_filter_change methods
    for neuron_folder in selected_neuron_folders:
        neuron_path = os.path.join(abs_filter_change_root, neuron_folder)
        for sub_folder in os.listdir(neuron_path):
            sub_folder_path = os.path.join(neuron_path, sub_folder)
            for degradation_folder in os.listdir(sub_folder_path):
                degradation_path = os.path.join(sub_folder_path, degradation_folder)
                save_degradation_folder = os.path.join(save_abs_filter_change_gradient_root, neuron_folder, sub_folder,
                                                       degradation_folder)
                os.makedirs(save_degradation_folder, exist_ok=True)

                img_list = sorted(glob.glob(os.path.join(degradation_path, '*.png')))
                for img_path in img_list:
                    imgname = os.path.splitext(os.path.basename(img_path))[0]
                    img = cv2.imread(img_path, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.
                    img = np.repeat(img, 3, axis=2)
                    img = img2tensor(img).unsqueeze(0).to(device)

                    with torch.no_grad():
                        gradient_map = grad_net(img).squeeze(0)

                    gradient_map = gradient_map.detach().cpu().numpy()

                    # save the gradient map
                    save_img_path = f'{save_degradation_folder}/{imgname}.npy'
                    np.save(save_img_path, gradient_map)

    # generate the gradient map of the output of random method
    # for neuron_folder in os.listdir(random_root):
    for neuron_folder in selected_neuron_folders:
        neuron_path = os.path.join(random_root, neuron_folder)
        for sub_folder in os.listdir(neuron_path):
            sub_folder_path = os.path.join(neuron_path, sub_folder)
            for degradation_folder in os.listdir(sub_folder_path):
                degradation_path = os.path.join(sub_folder_path, degradation_folder)
                save_degradation_folder = os.path.join(save_random_gradient_root, neuron_folder, sub_folder,
                                                       degradation_folder)
                os.makedirs(save_degradation_folder, exist_ok=True)

                img_list = sorted(glob.glob(os.path.join(degradation_path, '*.png')))
                for img_path in img_list:
                    imgname = os.path.splitext(os.path.basename(img_path))[0]
                    img = cv2.imread(img_path, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.
                    img = np.repeat(img, 3, axis=2)
                    img = img2tensor(img).unsqueeze(0).to(device)

                    with torch.no_grad():
                        gradient_map = grad_net(img).squeeze(0)

                    gradient_map = gradient_map.detach().cpu().numpy()

                    # save the gradient map
                    save_img_path = f'{save_degradation_folder}/{imgname}.npy'
                    np.save(save_img_path, gradient_map)
