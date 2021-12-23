import argparse
import cv2
import glob
import numpy as np
import os
import torch
from tqdm import tqdm

from archs.srcnn_style_arch import srcnn_style_net
from basicsr.utils.img_util import img2tensor


def faig(img1, img2, gt_img, baseline_model_path, target_model_path, total_step, conv_index):
    """ Filter Attribution Integrated Gradients of a single image

        When finding blurry filters, img1 is a blurry image,
            while img2 is a noisy image.
        When finding noisy filters, img1 is a noisy image,
            while img2 is a blurry image.

    Args:
        img1 (tensor): with the shape (1, 3, H, W)
        img2 (tensor): with the shape (1, 3, H, W)
        gt_img (tensor): with the shape (1, 3, H, W)
        baseline_model_path (str): path of baseline model
        target_model_path (str): path of target model
        total_step (int): total steps in the approximation of the integral
        conv_index (list): index of conv layer in srcnn-style like network

    Returns:
        faig_img1: faig result of img1
    """

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    baseline_state_dict = torch.load(baseline_model_path)['params_ema']
    target_state_dict = torch.load(target_model_path)['params_ema']

    # calculate the gradient of two images with different degradation
    total_gradient_img1 = 0
    total_gradient_img2 = 0

    # approximate the integral via 100 discrete points uniformly
    # sampled along the straight-line path
    for step in range(0, total_step):
        alpha = step / total_step
        interpolate_net_state_dict = {}
        for key, _ in baseline_state_dict.items():
            # a straight-line path between baseline model and target model
            interpolate_net_state_dict[key] = alpha * baseline_state_dict[key] + (1 - alpha) * target_state_dict[key]

        interpolate_net = srcnn_style_net(scale=2)
        interpolate_net.eval()
        interpolate_net = interpolate_net.to(device)
        interpolate_net.load_state_dict(interpolate_net_state_dict)

        # for degradation 1
        interpolate_net.zero_grad()
        output1 = interpolate_net(img1)
        # measure the distance between the network output and the ground-truth
        # refer to the equation 3 in the main paper
        criterion = torch.nn.MSELoss(reduction='sum')
        loss1 = criterion(gt_img, output1)
        # calculate the gradient of F to every filter
        loss1.backward()
        grad_list_img1 = []
        for idx in conv_index:
            grad = interpolate_net.features[idx].weight.grad
            grad = grad.reshape(-1, 3, 3)
            grad_list_img1.append(grad)
        grad_list_img1 = torch.cat(grad_list_img1, dim=0)
        total_gradient_img1 += grad_list_img1

        # for degradation 2
        interpolate_net.zero_grad()
        output2 = interpolate_net(img2)
        # measure the distance between the network output and the ground-truth
        # refer to the equation 3 in the main paper
        loss2 = criterion(gt_img, output2)
        # calculate the gradient of F to every filter
        loss2.backward()
        grad_list_img2 = []
        for idx in conv_index:
            grad = interpolate_net.features[idx].weight.grad
            grad = grad.reshape(-1, 3, 3)
            grad_list_img2.append(grad)
        grad_list_img2 = torch.cat(grad_list_img2, dim=0)
        total_gradient_img2 += grad_list_img2

    # calculate the diff of filters between the baseline model and target model
    diff_list = []
    baseline_net = srcnn_style_net(scale=2)
    baseline_net.eval()
    baseline_net = baseline_net.to(device)
    baseline_net.load_state_dict(baseline_state_dict)

    target_net = srcnn_style_net(scale=2)
    target_net.eval()
    target_net = target_net.to(device)
    target_net.load_state_dict(target_state_dict)
    for idx in conv_index:
        variation = baseline_net.features[idx].weight.detach() - target_net.features[idx].weight.detach()
        variation = variation.reshape(-1, 3, 3)
        diff_list.append(variation)
    diff_list = torch.cat(diff_list, dim=0)

    # multiple the cumulated gradients of img1 with the diff
    # refer to equation 6 in the main paper
    single_faig_img1 = total_gradient_img1 * diff_list / total_step
    single_faig_img1 = torch.sum(torch.sum(abs(single_faig_img1), dim=1), dim=1)

    # multiple the cumulated gradients of img2 with the diff
    # refer to equation 6 in the main paper
    single_faig_img2 = total_gradient_img2 * diff_list / total_step
    single_faig_img2 = torch.sum(torch.sum(abs(single_faig_img2), dim=1), dim=1)

    # Find discriminative filters for a specific degradation
    # refer to equation 7 in the main paper
    faig_img1 = single_faig_img1 - single_faig_img2
    return faig_img1.cpu().numpy()


def main():

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
    parser.add_argument('--gt_folder', type=str, default='datasets/Set14/GTmod12', help='folder that contains gt image')
    parser.add_argument(
        '--blur_folder', type=str, default='datasets/Set14/Blur2_LRbicx2', help='folder that contains blurry image')
    parser.add_argument(
        '--noise_folder', type=str, default='datasets/Set14/LRbicx2_noise0.1', help='folder that contains noisy image')
    parser.add_argument('--total_step', type=int, default=100)
    parser.add_argument('--conv_index', type=list, default=[0, 2, 4, 6, 8, 10, 12, 15, 17], help='index of conv layer')
    parser.add_argument(
        '--record_filters_folder',
        type=str,
        default='results/Interpret/neuron-search/srcnn_style/Set14/faig',
        help='folder that saves the sorted location index of discovered filters')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # configuration
    baseline_model_path = args.baseline_model_path
    target_model_path = args.target_model_path
    gt_folder = args.gt_folder
    blur_folder = args.blur_folder
    noise_folder = args.noise_folder
    total_step = args.total_step
    conv_index = args.conv_index

    record_filters_folder = args.record_filters_folder
    os.makedirs(record_filters_folder, exist_ok=True)

    noise_img_list = sorted(glob.glob(os.path.join(noise_folder, '*')))
    blurry_img_list = sorted(glob.glob(os.path.join(blur_folder, '*')))

    # deal noisy imgs
    # average all the gradient difference in a whole dataset
    faig_average_noisy = 0.0
    print('Now we sort the filters for noise!')
    pbar = tqdm(total=len(noise_img_list), desc='')
    for img_idx, path in enumerate(noise_img_list):
        imgname = os.path.basename(path)
        basename, _ = os.path.splitext(imgname)
        pbar.set_description(f'Read {basename}')
        pbar.update(1)

        noisy_img = cv2.imread(path, cv2.IMREAD_COLOR).astype(np.float32) / 255.
        noisy_img = img2tensor(noisy_img).unsqueeze(0).to(device)
        blurry_img = cv2.imread(blurry_img_list[img_idx], cv2.IMREAD_COLOR).astype(np.float32) / 255.
        blurry_img = img2tensor(blurry_img).unsqueeze(0).to(device)
        gt_img_path = os.path.join(gt_folder, imgname)
        gt_img = cv2.imread(gt_img_path, cv2.IMREAD_COLOR).astype(np.float32) / 255.
        gt_img = img2tensor(gt_img).unsqueeze(0).to(device)

        # use faig for a single image
        faig_noisy = faig(noisy_img, blurry_img, gt_img, baseline_model_path, target_model_path, total_step, conv_index)
        faig_average_noisy += np.array(faig_noisy)
    faig_average_noisy /= len(noise_img_list)
    # sort the neurons in descending order
    sorted_noisy_location = np.argsort(faig_average_noisy)[::-1]
    save_noisy_filter_txt = os.path.join(record_filters_folder, 'noise_index.txt')
    np.savetxt(save_noisy_filter_txt, sorted_noisy_location, delimiter=',', fmt='%d')
    pbar.close()

    # deal blurry imgs
    # average all the gradient difference in a whole dataset
    faig_average_blurry = 0.0
    print('Now we sort the filters for blur!')
    pbar = tqdm(total=len(blurry_img_list), desc='')
    for img_idx, path in enumerate(blurry_img_list):
        imgname = os.path.basename(path)
        basename, _ = os.path.splitext(imgname)
        pbar.set_description(f'Read {basename}')
        pbar.update(1)

        blurry_img = cv2.imread(path, cv2.IMREAD_COLOR).astype(np.float32) / 255.
        blurry_img = img2tensor(blurry_img).unsqueeze(0).to(device)
        noisy_img = cv2.imread(noise_img_list[img_idx], cv2.IMREAD_COLOR).astype(np.float32) / 255.
        noisy_img = img2tensor(noisy_img).unsqueeze(0).to(device)
        gt_img_path = os.path.join(gt_folder, imgname)
        gt_img = cv2.imread(gt_img_path, cv2.IMREAD_COLOR).astype(np.float32) / 255.
        gt_img = img2tensor(gt_img).unsqueeze(0).to(device)

        # use faig for a single image
        faig_blurry = faig(blurry_img, noisy_img, gt_img, baseline_model_path, target_model_path, total_step,
                           conv_index)
        faig_average_blurry += np.array(faig_blurry)
    faig_average_blurry /= len(blurry_img_list)
    # sort the neurons in descending order
    sorted_blurry_location = np.argsort(faig_average_blurry)[::-1]
    save_blurry_filter_txt = os.path.join(record_filters_folder, 'blur_index.txt')
    np.savetxt(save_blurry_filter_txt, sorted_blurry_location, delimiter=',', fmt='%d')
    pbar.close()


if __name__ == '__main__':
    main()
