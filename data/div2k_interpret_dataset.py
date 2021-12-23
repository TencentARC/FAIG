import cv2
import numpy as np
import os.path as osp
import torch.utils.data as data
from torchvision.transforms.functional import normalize

from basicsr.data.data_util import paths_from_lmdb
from basicsr.data.degradations import bivariate_Gaussian
from basicsr.data.transforms import augment, paired_random_crop
from basicsr.utils import FileClient, get_root_logger, imfrombytes, img2tensor, scandir
from basicsr.utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class DIV2KInterpretDataset(data.Dataset):
    """Dataset specified for Interpreting Blind SR Network.

    Read GT images and then generate degraded LQ images on-the-fly.
    Now it just supports single degradation type.
    """

    def __init__(self, opt):
        super(DIV2KInterpretDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None

        self.gt_folder = opt['dataroot_gt']

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.gt_folder]
            self.io_backend_opt['client_keys'] = ['gt']
            self.paths = paths_from_lmdb(self.gt_folder)
        elif 'meta_info_file' in self.opt:
            with open(self.opt['meta_info_file'], 'r') as fin:
                self.paths = [osp.join(self.gt_folder, line.split(' ')[0]) for line in fin]
        else:
            self.paths = sorted(list(scandir(self.gt_folder, full_path=True)))

        self.kernel_prob = opt['kernel_prob']
        self.kernel_size = opt['kernel_size']
        self.sigma_list = opt['sigma_list']

        self.noise_prob = opt['noise_prob']
        self.noise_list = opt['noise_list']

        logger = get_root_logger()
        if self.kernel_prob != 0.0:
            logger.info(f'kernel_prob: {self.kernel_prob}')
            logger.info(f'Kernel size: {self.kernel_size}')
            logger.info(f'sigma: [{", ".join(map(str, self.sigma_list))}]')
        if self.noise_prob != 0.0:
            logger.info(f'noise_prob: {self.noise_prob}')
            logger.info(f'noise: [{", ".join(map(str, self.noise_list))}]')

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']
        # Load gt images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path = self.paths[index]
        img_bytes = self.file_client.get(gt_path, 'gt')
        img_gt = imfrombytes(img_bytes, float32=True)
        h, w, _ = img_gt.shape

        # blur
        img_lq = img_gt.copy()
        if np.random.uniform() < self.kernel_prob:
            sigma_option = len(self.sigma_list)
            sigma = self.sigma_list[np.random.choice(sigma_option)]
            kernel = bivariate_Gaussian(self.kernel_size, sigma, sigma, theta=0, isotropic=True)
            img_lq = cv2.filter2D(img_lq, -1, kernel)

        # downsample
        img_lq = cv2.resize(img_lq, (w // scale, h // scale), cv2.INTER_CUBIC)

        # noise
        if np.random.uniform() < self.noise_prob:
            noise_option = len(self.noise_list)
            noise_sigma = self.noise_list[np.random.choice(noise_option)]
            noise = np.float32(np.random.randn(*(img_lq.shape))) * noise_sigma
            img_lq = img_lq + noise

        # round and clip
        img_lq = np.clip((img_lq * 255.0).round(), 0, 255) / 255.

        # augmentation for training
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            # random crop
            img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale, gt_path)
            # flip, rotation
            img_gt, img_lq = augment([img_gt, img_lq], self.opt['use_flip'], self.opt['use_rot'])

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)

        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_gt, self.mean, self.std, inplace=True)
            normalize(img_lq, self.mean, self.std, inplace=True)

        return {'gt': img_gt, 'lq': img_lq, 'gt_path': gt_path}

    def __len__(self):
        return len(self.paths)
