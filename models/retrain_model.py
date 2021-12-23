import numpy as np
import torch
from collections import OrderedDict
from copy import deepcopy
from os import path as osp
from torch.nn import functional as F
from tqdm import tqdm

from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.metrics import calculate_metric
from basicsr.models.base_model import BaseModel
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.registry import MODEL_REGISTRY


@MODEL_REGISTRY.register()
class ReTrainModel(BaseModel):
    """Base SR model for single image super-resolution."""

    def __init__(self, opt):
        super(ReTrainModel, self).__init__(opt)

        # define network net_g
        self.net_g = build_network(deepcopy(opt['network_g']))
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)
        # load pretrained model
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            self.load_network(self.net_g, load_path, self.opt['path'].get('strict_load_g', True))

        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema only used for testing on one GPU and saving.
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = build_network(deepcopy(self.opt['network_g'])).to(self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        # define losses
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
        else:
            self.cri_pix = None

        if train_opt.get('perceptual_opt'):
            self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None

        if self.cri_pix is None and self.cri_perceptual is None:
            raise ValueError('Both pixel and perceptual losses are None.')

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']

        optim_params = []
        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')

        optim_type = train_opt['optim_g'].pop('type')
        if optim_type == 'Adam':
            self.optimizer_g = torch.optim.Adam(optim_params, **train_opt['optim_g'])
        else:
            raise NotImplementedError(f'optimizer {optim_type} is not supperted yet.')
        self.optimizers.append(self.optimizer_g)

        # just update these neurons
        # TODO: Now we only support for re-training the discovered filters within srcnn_style network
        finetuned_neuron_index_file = train_opt.get('finetuned_neuron_index_file', [])
        finetuned_neurons = list(np.loadtxt(finetuned_neuron_index_file, dtype=int))
        neuron_ratio = float(train_opt['neuron_ratio'])
        finetuned_neurons = finetuned_neurons[:int(len(finetuned_neurons) * neuron_ratio)]

        cumulate_num_neurons = [192, 8384, 24768, 41152, 57536, 73920, 139456, 155840, 156224]
        conv_index = [0, 2, 4, 6, 8, 10, 12, 15, 17]
        # calculate the location of these neurons
        layer_neuron = []
        row_neuron = []
        column_neuron = []
        for neuron_index in finetuned_neurons:
            if neuron_index < 192:
                layer = 0
                row = neuron_index // 3
                column = neuron_index % 3
            else:
                for i in range(1, len(cumulate_num_neurons)):
                    if neuron_index < cumulate_num_neurons[i]:
                        layer = conv_index[i]
                        row = (neuron_index -
                               cumulate_num_neurons[i - 1]) // self.net_g.module.features[layer].weight.size(1)
                        column = (neuron_index -
                                  cumulate_num_neurons[i - 1]) % self.net_g.module.features[layer].weight.size(1)
                        layer_neuron.append(layer)
                        row_neuron.append(row)
                        column_neuron.append(column)
                        break

        # generate the updated params mask
        self.mask_state_dict = {}
        for k, v in self.net_g.named_parameters():
            self.mask_state_dict[k] = torch.zeros_like(v)

        for idx, layer in enumerate(layer_neuron):
            self.mask_state_dict[f'module.features.{layer}.weight'][
                row_neuron[idx], column_neuron[idx], :, :] = torch.ones_like(
                    self.mask_state_dict[f'module.features.{layer}.weight'][row_neuron[idx], column_neuron[idx], :, :])

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

    def optimize_parameters(self, current_iter):

        self.optimizer_g.zero_grad()
        original_state_dict = {}
        for k, v in self.net_g.named_parameters():
            original_state_dict[k] = v.clone()
        self.output = self.net_g(self.lq)

        l_total = 0
        loss_dict = OrderedDict()
        # pixel loss
        if self.cri_pix:
            l_pix = self.cri_pix(self.output, self.gt)
            l_total += l_pix
            loss_dict['l_pix'] = l_pix
        # perceptual loss
        if self.cri_perceptual:
            l_percep, l_style = self.cri_perceptual(self.output, self.gt)
            if l_percep is not None:
                l_total += l_percep
                loss_dict['l_percep'] = l_percep
            if l_style is not None:
                l_total += l_style
                loss_dict['l_style'] = l_style

        l_total.backward()
        self.optimizer_g.step()

        updated_state_dict = {}
        for k, v in self.net_g.named_parameters():
            updated_state_dict[k] = v.clone()

        new_state_dict = {}
        # mention that we just update the selected neurons
        for key, _ in original_state_dict.items():
            new_state_dict[key] = (1 - self.mask_state_dict[key]) * original_state_dict[key].clone(
            ) + self.mask_state_dict[key] * updated_state_dict[key].clone()

        self.net_g.load_state_dict(new_state_dict, strict=True)

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def test(self):
        # mod crop
        mod_scale = self.opt.get('mod_scale')
        if mod_scale is not None:
            h_pad, w_pad = 0, 0
            _, _, h, w = self.lq.size()
            if (h % mod_scale != 0):
                h_pad = (mod_scale - h % mod_scale)
            if (w % mod_scale != 0):
                w_pad = (mod_scale - w % mod_scale)
            self.lq = F.pad(self.lq, (0, w_pad, 0, h_pad), 'reflect')

        if self.ema_decay > 0:
            self.net_g_ema.eval()
            with torch.no_grad():
                self.output = self.net_g_ema(self.lq)
        else:
            self.net_g.eval()
            with torch.no_grad():
                self.output = self.net_g(self.lq)
            self.net_g.train()

        # remove extra pad
        if mod_scale is not None:
            _, _, h, w = self.output.size()
            self.output = self.output[:, :, 0:h - h_pad, 0:w - w_pad]

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        if self.opt['rank'] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
        pbar = tqdm(total=len(dataloader), unit='image')

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data(val_data)
            self.test()

            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals['result']])
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']])
                del self.gt

            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()

            if save_img:
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'], img_name,
                                             f'{img_name}_{current_iter}.png')
                else:
                    if self.opt['val']['suffix']:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["val"]["suffix"]}.png')
                    else:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["name"]}.png')
                imwrite(sr_img, save_img_path)

            if with_metrics:
                # calculate metrics
                opt_metric = deepcopy(self.opt['val']['metrics'])
                for name, opt_ in opt_metric.items():
                    metric_data = dict(img1=sr_img, img2=gt_img)
                    self.metric_results[name] += calculate_metric(metric_data, opt_)
            pbar.update(1)
            pbar.set_description(f'Test {img_name}')
        pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}\n'
        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{metric}', value, current_iter)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        if self.ema_decay > 0:
            self.save_network([self.net_g, self.net_g_ema], 'net_g', current_iter, param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)
