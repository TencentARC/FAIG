import argparse
import cv2
import glob
import numpy as np
import os
import torch
from tqdm import tqdm

from basicsr.archs.srresnet_arch import MSRResNet
from basicsr.utils.img_util import img2tensor


def faig(img1, img2, gt_img, baseline_model_path, target_model_path, total_step, conv_name_list, scale):
    """ filter Attribution Integrated Gradients.

        When finding blurry neurons, img1 is a blurry image,
            while img2 is a noisy image.
        When finding noisy neurons, img1 is a noisy image,
            while img2 is a blurry image.

    Args:
        img1 (tensor): with the shape (1, 3, H, W)
        img2 (tensor): with the shape (1, 3, H, W)
        gt_img (tensor): with the shape (1, 3, H, W)
        baseline_model_path: path of baseline model
        target_model_path: path of target model
        total_step (int): bisection of partition
        conv_name_list (list)
        scale (int)

    Returns:
        sorted_diff (list): sorted values
        sorted_index (list): sorted index of kernel
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
        # define current net
        alpha = step / total_step
        current_net_state_dict = {}
        for key, _ in baseline_state_dict.items():
            # a straight-line path between baseline model and target model
            current_net_state_dict[key] = alpha * baseline_state_dict[key] + (1 - alpha) * target_state_dict[key]

        current_net = MSRResNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=16, upscale=scale)
        current_net.eval()
        current_net = current_net.to(device)
        current_net.load_state_dict(current_net_state_dict)

        # for degradation 1
        current_net.zero_grad()
        output1 = current_net(img1)
        # measure the distance between the network output and the ground-truth
        # refer to the equation 3 in the main paper
        criterion = torch.nn.MSELoss(reduction='sum')
        loss1 = criterion(gt_img, output1)
        # calculate the gradient of F to every filter
        loss1.backward()
        grad_list_img1 = []
        # save the gradient of all filters to grad_list_img1
        # add conv_first
        grad = current_net.conv_first.weight.grad
        grad = grad.reshape(-1, 3, 3)
        grad_list_img1.append(grad)

        # add body module
        for i in range(16):
            grad = current_net.body[i].conv1.weight.grad
            grad = grad.reshape(-1, 3, 3)
            grad_list_img1.append(grad)

            grad = current_net.body[i].conv2.weight.grad
            grad = grad.reshape(-1, 3, 3)
            grad_list_img1.append(grad)

        # add upconv1
        grad = current_net.upconv1.weight.grad
        grad = grad.reshape(-1, 3, 3)
        grad_list_img1.append(grad)

        # add upconv2
        if scale == 4:
            grad = current_net.upconv2.weight.grad
            grad = grad.reshape(-1, 3, 3)
            grad_list_img1.append(grad)

        # add conv_hr
        grad = current_net.conv_hr.weight.grad
        grad = grad.reshape(-1, 3, 3)
        grad_list_img1.append(grad)

        # add conv_last
        grad = current_net.conv_last.weight.grad
        grad = grad.reshape(-1, 3, 3)
        grad_list_img1.append(grad)

        # reshape to [-1, 3, 3]
        grad_list_img1 = torch.cat(grad_list_img1, dim=0)
        total_gradient_img1 += grad_list_img1

        # Input img2
        current_net.zero_grad()
        output2 = current_net(img2)

        loss2 = criterion(gt_img, output2)
        # calculate the gradient of F to every filter
        loss2.backward()
        grad_list_img2 = []
        # save all grad to list
        # add conv_first
        grad = current_net.conv_first.weight.grad
        grad = grad.reshape(-1, 3, 3)
        grad_list_img2.append(grad)

        # add body module
        for i in range(16):
            grad = current_net.body[i].conv1.weight.grad
            grad = grad.reshape(-1, 3, 3)
            grad_list_img2.append(grad)

            grad = current_net.body[i].conv2.weight.grad
            grad = grad.reshape(-1, 3, 3)
            grad_list_img2.append(grad)

        # add upconv1
        grad = current_net.upconv1.weight.grad
        grad = grad.reshape(-1, 3, 3)
        grad_list_img2.append(grad)

        # add upconv2
        if scale == 4:
            grad = current_net.upconv2.weight.grad
            grad = grad.reshape(-1, 3, 3)
            grad_list_img2.append(grad)

        # add conv_hr
        grad = current_net.conv_hr.weight.grad
        grad = grad.reshape(-1, 3, 3)
        grad_list_img2.append(grad)

        # add conv_last
        grad = current_net.conv_last.weight.grad
        grad = grad.reshape(-1, 3, 3)
        grad_list_img2.append(grad)

        # reshape to [-1, 3, 3]
        grad_list_img2 = torch.cat(grad_list_img2, dim=0)
        total_gradient_img2 += grad_list_img2

    # multiple the variation
    diff_list = []
    for key in conv_name_list:
        variation = baseline_state_dict[key] - target_state_dict[key]
        variation = variation.reshape(-1, 3, 3)
        diff_list.append(variation)
    diff_list = torch.cat(diff_list, dim=0).to(device)

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
        default='experiments/srresnet/baseline_model.pth',
        help='path of baseline model')
    parser.add_argument(
        '--target_model_path', type=str, default='experiments/srresnet/target_model.pth', help='path of target model')
    parser.add_argument('--gt_folder', type=str, default='datasets/Set14/GTmod12', help='folder that contains gt image')
    parser.add_argument(
        '--blur_folder', type=str, default='datasets/Set14/Blur2_LRbicx2', help='folder that contains blurry image')
    parser.add_argument(
        '--noise_folder', type=str, default='datasets/Set14/LRbicx2_noise0.1', help='folder that contains noisy image')
    parser.add_argument('--total_step', type=int, default=100)
    parser.add_argument('--scale', type=int, default=2, help='scale ratio')
    parser.add_argument(
        '--record_filters_folder',
        type=str,
        default='results/Interpret/neuron-search/srresnet/Set14/faig',
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
    scale = args.scale

    record_filters_folder = args.record_filters_folder
    os.makedirs(record_filters_folder, exist_ok=True)

    conv_name_list = []
    cumulate_num_neurons = [0]
    target_state_dict = torch.load(target_model_path)['params_ema']
    # Note that we exclude bias
    for key, value in target_state_dict.items():
        if key.find('weight') != -1:
            conv_name_list.append(key)
            num_neurons = value.size(0) * value.size(1)
            cumulate_num_neurons.append(cumulate_num_neurons[-1] + num_neurons)
    # del the first element in cumulate_num_neurons
    del cumulate_num_neurons[0]

    noise_img_list = sorted(glob.glob(os.path.join(noise_folder, '*')))
    blur_img_list = sorted(glob.glob(os.path.join(blur_folder, '*')))

    # deal noisy imgs
    # average all the gradient difference in a whole dataset
    faig_average_noisy = 0.0
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
        blurry_img = cv2.imread(blur_img_list[img_idx], cv2.IMREAD_COLOR).astype(np.float32) / 255.
        blurry_img = img2tensor(blurry_img).unsqueeze(0).to(device)
        gt_img_path = os.path.join(gt_folder, imgname)
        gt_img = cv2.imread(gt_img_path, cv2.IMREAD_COLOR).astype(np.float32) / 255.
        gt_img = img2tensor(gt_img).unsqueeze(0).to(device)

        # calculate the neurons
        diff = faig(
            noisy_img,
            blurry_img,
            gt_img,
            baseline_model_path,
            target_model_path,
            total_step,
            conv_name_list,
            scale=scale)
        faig_average_noisy += np.array(diff)

    # sort the neurons in descending order
    sorted_noisy_location = np.argsort(faig_average_noisy)[::-1]
    save_noisy_filter_txt = os.path.join(record_filters_folder, 'noise_index.txt')
    np.savetxt(save_noisy_filter_txt, sorted_noisy_location, delimiter=',', fmt='%d')
    pbar.close()

    # deal blurry imgs
    # average all the gradient difference in a whole dataset
    faig_average_blurry = 0.0
    print('Now we sort the filters for blur!')
    pbar = tqdm(total=len(blur_img_list), desc='')
    for img_idx, path in enumerate(blur_img_list):
        # read image
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

        # calculate the neurons
        diff = faig(
            blurry_img,
            noisy_img,
            gt_img,
            baseline_model_path,
            target_model_path,
            total_step,
            conv_name_list,
            scale=scale)
        faig_average_blurry += np.array(diff)

    sorted_blurry_location = np.argsort(faig_average_blurry)[::-1]
    save_blurry_filter_txt = os.path.join(record_filters_folder, 'blur_index.txt')
    np.savetxt(save_blurry_filter_txt, sorted_blurry_location, delimiter=',', fmt='%d')
    pbar.close()


if __name__ == '__main__':
    main()
