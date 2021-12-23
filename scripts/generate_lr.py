import argparse
import cv2
import numpy as np
import os
from tqdm import tqdm

from basicsr.data.degradations import bivariate_Gaussian
from basicsr.utils import scandir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ori_folder', type=str, default='datasets/Set14/GTmod12', help='hr image folder')
    parser.add_argument(
        '--save_noise_folder', type=str, default='datasets/Set14/LRbicx2_noise0.1', help='save noise folder')
    parser.add_argument('--save_blur_folder', type=str, default='datasets/Set14/Blur2_LRbicx2', help='save blur folder')
    parser.add_argument('--kernel_size', type=int, default=21)
    parser.add_argument('--kernel_sigma', type=int, default=2)
    parser.add_argument('--noise_sigma', type=float, default=0.1)
    args = parser.parse_args()

    ori_folder = args.ori_folder
    save_noise_folder = args.save_noise_folder
    save_blur_folder = args.save_blur_folder

    os.makedirs(save_noise_folder, exist_ok=True)
    os.makedirs(save_blur_folder, exist_ok=True)

    img_list = sorted(list(scandir(ori_folder, full_path=True)))

    pbar = tqdm(total=len(img_list), desc='')
    for img_path in img_list:
        basename = os.path.basename(img_path)
        pbar.update(1)
        pbar.set_description(f'Read {basename}')
        img = cv2.imread(img_path) / 255.0
        h, w, _ = img.shape

        # generate blurry image
        kernel_size = args.kernel_size
        kernel_sigma = args.kernel_sigma
        blur_kernel = bivariate_Gaussian(kernel_size, kernel_sigma, kernel_sigma, theta=0.0, isotropic=True)
        blur_img = cv2.filter2D(img, -1, blur_kernel)
        blur_img = cv2.resize(blur_img, (w // 2, h // 2), interpolation=cv2.INTER_LINEAR)
        blur_img = np.clip((blur_img * 255.0).round(), 0, 255) / 255.
        save_blur_img_path = os.path.join(save_blur_folder, basename)
        cv2.imwrite(save_blur_img_path, blur_img * 255.0)

        # generate noisy image
        img = cv2.resize(img, (w // 2, h // 2), interpolation=cv2.INTER_LINEAR)
        # add gaussian noise
        noise_sigma = args.noise_sigma
        noise = np.float32(np.random.randn(*(img.shape))) * noise_sigma
        noise_img = img + noise
        noise_img = np.clip((noise_img * 255.0).round(), 0, 255) / 255.
        save_noise_img_path = os.path.join(save_noise_folder, basename)
        cv2.imwrite(save_noise_img_path, noise_img * 255.0)


if __name__ == '__main__':
    main()
