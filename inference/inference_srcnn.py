import argparse
import cv2
import glob
import numpy as np
import os
import torch
from tqdm import tqdm

import basicsr.metrics.psnr_ssim as psnr_ssim
from archs.srcnn_style_arch import srcnn_style_net
from basicsr.utils import img2tensor, tensor2img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='experiments/srcnn_style/target_model.pth')
    parser.add_argument('--dataroot', type=str, default='datasets/Set14', help='gt image folder')
    parser.add_argument('--save_root', type=str, default='results/Interpret/srcnn-targetmodel-sr', help='save folder')
    args = parser.parse_args()

    os.makedirs(args.save_root, exist_ok=True)

    test_folders = ['Blur2_LRbicx2', 'LRbicx2_noise0.1']
    gt_folder = os.path.join(args.dataroot, 'GTmod12')

    print(args.model_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # set up model

    model = srcnn_style_net(scale=2)
    model.load_state_dict(torch.load(args.model_path)['params'], strict=True)
    model.eval()
    model = model.to(device)

    for test_folder in test_folders:
        test_root = os.path.join(args.dataroot, test_folder)

        img_list = sorted(glob.glob(os.path.join(test_root, '*')))
        pbar = tqdm(total=len(img_list), desc='')

        save_folder = os.path.join(args.save_root, test_folder)
        os.makedirs(save_folder, exist_ok=True)

        avg_psnr = 0.0
        for path in img_list:
            imgname = os.path.splitext(os.path.basename(path))[0]
            pbar.update(1)
            pbar.set_description(f'Read image {imgname}')
            # read image
            img = cv2.imread(path, cv2.IMREAD_COLOR).astype(np.float32) / 255.
            img = img2tensor(img)
            img = img.unsqueeze(0).to(device)
            # inference
            with torch.no_grad():
                output = model(img).squeeze(0)

            # save image
            output = tensor2img(output, min_max=(0, 1))

            # calculate psnr
            gt_img_path = os.path.join(gt_folder, f'{imgname}.png')
            gt_img = cv2.imread(gt_img_path, cv2.IMREAD_COLOR).astype(np.float32)

            crt_psnr = psnr_ssim.calculate_psnr(output, gt_img, crop_border=2)
            avg_psnr += crt_psnr

            cv2.imwrite(f'{save_folder}/{imgname}.png', output)
        pbar.close()

        avg_psnr /= len(img_list)
        print(f'psnr is: {avg_psnr:.6f}')


if __name__ == '__main__':
    main()
