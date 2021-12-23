import argparse
import cv2
import glob
import numpy as np
import os
import torch
from tqdm import tqdm

from basicsr.archs.srresnet_arch import MSRResNet
from basicsr.utils.img_util import img2tensor


def ig(base_img, final_img, finetune_state_dict, total_step, scale):
    """
    Args:
        base_img (tensor): with the shape (1, 3, H, W)
        final_img: (tensor): with the shape (1, 3, H, W)
        finetune_state_dict: state_dict of finetuned net

    Returns:
        sorted_diff (list): sorted values
        sorted_index (list): sorted index of kernel
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    total_gradient = 0

    finetune_net = MSRResNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=16, upscale=scale)
    finetune_net.eval()
    finetune_net = finetune_net.to(device)
    finetune_net.load_state_dict(finetune_state_dict)

    for step in range(0, total_step):
        alpha = step / total_step
        # define interpolated img
        interpolated_img = base_img + alpha * (final_img - base_img)
        finetune_net.zero_grad()
        interpolated_output = finetune_net(interpolated_img)

        loss = interpolated_output.sum()
        loss.backward()
        grad_list = []
        # calculate the gradient of conv
        # add conv_fist
        grad = finetune_net.conv_first.weight.grad
        grad = grad.reshape(-1, 3, 3)
        grad_list.append(grad)

        # add body module
        for i in range(16):
            grad = finetune_net.body[i].conv1.weight.grad
            grad = grad.reshape(-1, 3, 3)
            grad_list.append(grad)

            grad = finetune_net.body[i].conv2.weight.grad
            grad = grad.reshape(-1, 3, 3)
            grad_list.append(grad)

        # add upconv1
        grad = finetune_net.upconv1.weight.grad
        grad = grad.reshape(-1, 3, 3)
        grad_list.append(grad)

        # add upconv2
        if scale == 4:
            grad = finetune_net.upconv2.weight.grad
            grad = grad.reshape(-1, 3, 3)
            grad_list.append(grad)

        # add conv_hr
        grad = finetune_net.conv_hr.weight.grad
        grad = grad.reshape(-1, 3, 3)
        grad_list.append(grad)

        # add conv_last
        grad = finetune_net.conv_last.weight.grad
        grad = grad.reshape(-1, 3, 3)
        grad_list.append(grad)

        # [-1, 3, 3]
        grad_list = torch.cat(grad_list, dim=0)

        total_gradient += grad_list

    total_diff = torch.sum(torch.sum(abs(total_gradient), dim=1), dim=1) / total_step

    # Note that we do not multiple (final_img - base_img)

    return total_diff.cpu().numpy()


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--target_model_path', type=str, default='experiments/srresnet/target_model.pth', help='path of target model')
    parser.add_argument(
        '--bic_folder', type=str, default='datasets/Set14/Linear_LRbicx2', help='folder that contains bicubic image')
    parser.add_argument(
        '--blur_folder', type=str, default='datasets/Set14/Blur2_LRbicx2', help='folder that contains blurry image')
    parser.add_argument(
        '--noise_folder', type=str, default='datasets/Set14/LRbicx2_noise0.1', help='folder that contains noisy image')
    parser.add_argument('--total_step', type=int, default=100)
    parser.add_argument('--scale', type=int, default=2, help='scale ratio')
    parser.add_argument(
        '--record_filters_folder',
        type=str,
        default='results/Interpret/neuron-search/srresnet/Set14/ig',
        help='folder that saves the sorted location index of discovered filters')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # configuration
    target_model_path = args.target_model_path
    bic_folder = args.bic_folder
    blur_folder = args.blur_folder
    noise_folder = args.noise_folder
    total_step = args.total_step
    scale = args.scale

    record_filters_folder = args.record_filters_folder
    os.makedirs(record_filters_folder, exist_ok=True)

    target_state_dict = torch.load(target_model_path)['params_ema']

    bic_img_list = sorted(glob.glob(os.path.join(bic_folder, '*')))
    blur_img_list = sorted(glob.glob(os.path.join(blur_folder, '*')))
    noise_img_list = sorted(glob.glob(os.path.join(noise_folder, '*')))

    # deal noisy imgs
    ig_average_noisy = 0.0
    print('Now we sort the filters for noise!')
    pbar = tqdm(total=len(blur_img_list), desc='')
    for img_idx, path in enumerate(noise_img_list):
        # read image
        imgname = os.path.basename(path)
        basename, _ = os.path.splitext(imgname)
        pbar.set_description(f'Read {basename}')
        pbar.update(1)

        noisy_img = cv2.imread(path, cv2.IMREAD_COLOR).astype(np.float32) / 255.
        noisy_img = img2tensor(noisy_img).unsqueeze(0).to(device)
        bic_img = cv2.imread(bic_img_list[img_idx], cv2.IMREAD_COLOR).astype(np.float32) / 255.
        bic_img = torch.from_numpy(np.transpose(bic_img[:, :, [2, 1, 0]], (2, 0, 1))).float()
        bic_img = bic_img.unsqueeze(0).to(device)

        # calculate the neurons
        diff = ig(bic_img, noisy_img, target_state_dict, total_step, scale=scale)
        ig_average_noisy += np.array(diff)

    sorted_noisy_location = np.argsort(ig_average_noisy)[::-1]
    save_noisy_filter_txt = os.path.join(record_filters_folder, 'noise_index.txt')
    np.savetxt(save_noisy_filter_txt, sorted_noisy_location, delimiter=',', fmt='%d')
    pbar.close()

    # deal blurry imgs
    ig_average_blurry = 0.0
    print('Now we sort the filters for blur!')
    pbar = tqdm(total=len(blur_img_list), desc='')
    for img_idx, path in enumerate(blur_img_list):
        # read image
        imgname = os.path.basename(path)
        basename, _ = os.path.splitext(imgname)
        pbar.set_description(f'Read {basename}')
        pbar.update(1)

        blurry_img = cv2.imread(path, cv2.IMREAD_COLOR).astype(np.float32) / 255.
        blurry_img = torch.from_numpy(np.transpose(blurry_img[:, :, [2, 1, 0]], (2, 0, 1))).float()
        blurry_img = blurry_img.unsqueeze(0).to(device)
        bic_img = cv2.imread(bic_img_list[img_idx], cv2.IMREAD_COLOR).astype(np.float32) / 255.
        bic_img = torch.from_numpy(np.transpose(bic_img[:, :, [2, 1, 0]], (2, 0, 1))).float()
        bic_img = bic_img.unsqueeze(0).to(device)

        # calculate the neurons
        diff = ig(bic_img, blurry_img, target_state_dict, total_step, scale=scale)
        ig_average_blurry += np.array(diff)

    sorted_blurry_location = np.argsort(ig_average_blurry)[::-1]
    save_blurry_filter_txt = os.path.join(record_filters_folder, 'blur_index.txt')
    np.savetxt(save_blurry_filter_txt, sorted_blurry_location, delimiter=',', fmt='%d')
    pbar.close()


if __name__ == '__main__':
    main()
