import argparse
import os
from mask_neurons import mask_neurons_with_different_proportion

from analysis.scripts.cal_blur_gradient_loss import calc_blur_gradient_loss
from analysis.scripts.cal_noise_gradient_loss import calc_noise_gradient_loss
from analysis.scripts.generate_gradient_map import generate_gradient_maps_within_several_folders


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--baseline_model_path',
        type=str,
        default='experiments/srresnet/baseline_model.pth',
        help='path of baseline model')
    parser.add_argument(
        '--target_model_path', type=str, default='experiments/srresnet/target_model.pth', help='path of target model')
    parser.add_argument('--scale', type=int, default=2, help='scale ratio')
    parser.add_argument('--dataset', type=str, default='Set14', help='dataset name')
    parser.add_argument('--net_type', type=str, default='srresnet', help='network type')
    parser.add_argument('--blur_folder_name', type=str, default='Blur2_LRbicx2', help='name of blurry folder')
    parser.add_argument('--noise_folder_name', type=str, default='LRbicx2_noise0.1', help='name of noisy folder')
    parser.add_argument(
        '--srresnet_neuron_folder',
        type=str,
        default='results/Interpret/neuron-search/srresnet',
        help='folder that contains the discvoered important neurons of different methods in SRResNet')
    parser.add_argument(
        '--srresnet_mask_save_folder',
        type=str,
        default='results/Interpret/masking/srresnet',
        help='folder that saves the masking results')
    parser.add_argument(
        '--srresnet_gradient_save_folder',
        type=str,
        default='results/Interpret/gradient/srresnet',
        help='folder that saves the gradient map')
    parser.add_argument(
        '--srresnet_metric_save_folder',
        type=str,
        default='results/Interpret/metrics/srresnet',
        help='folder that saves the gradient difference')
    parser.add_argument(
        '--target_model_output_root',
        type=str,
        default='results/Interpret/srresnet-targetmodel-sr',
        help='folder that contains the restored results of target model')
    args = parser.parse_args()

    baseline_model_path = args.baseline_model_path
    target_model_path = args.target_model_path
    scale = args.scale
    dataset = args.dataset
    net_type = args.net_type
    blur_folder_name = args.blur_folder_name
    noise_folder_name = args.noise_folder_name

    dataset_root = f'datasets/{dataset}'

    srresnet_neuron_folder = args.srresnet_neuron_folder
    srresnet_mask_save_folder = args.srresnet_mask_save_folder
    srresnet_gradient_save_folder = args.srresnet_gradient_save_folder
    srresnet_metric_save_folder = args.srresnet_metric_save_folder

    # faig
    faig_noisy_neuron_txt = f'{srresnet_neuron_folder}/{dataset}/faig/noise_index.txt'
    faig_blurry_neuron_txt = f'{srresnet_neuron_folder}/{dataset}/faig/blur_index.txt'
    save_mask_faig_filters_folder = f'{srresnet_mask_save_folder}/{dataset}/faig'
    os.makedirs(save_mask_faig_filters_folder, exist_ok=True)

    # ig
    ig_noisy_neuron_txt = f'{srresnet_neuron_folder}/{dataset}/ig/noise_index.txt'
    ig_blurry_neuron_txt = f'{srresnet_neuron_folder}/{dataset}/ig/blur_index.txt'
    save_mask_ig_filters_folder = f'{srresnet_mask_save_folder}/{dataset}/ig'
    os.makedirs(save_mask_ig_filters_folder, exist_ok=True)

    # abs_filter_change
    abs_filter_change_noisy_neuron_txt = f'{srresnet_neuron_folder}/{dataset}/abs_filter_change/noise_index.txt'
    abs_filter_change_blurry_neuron_txt = f'{srresnet_neuron_folder}/{dataset}/abs_filter_change/blur_index.txt'
    save_mask_abs_filter_change_filters_folder = f'{srresnet_mask_save_folder}/{dataset}/abs_filter_change'
    os.makedirs(save_mask_abs_filter_change_filters_folder, exist_ok=True)

    # random
    random_noisy_neuron_txt = f'{srresnet_neuron_folder}/{dataset}/random/noise_index.txt'
    random_blurry_neuron_txt = f'{srresnet_neuron_folder}/{dataset}/random/blur_index.txt'
    save_mask_random_filters_folder = f'{srresnet_mask_save_folder}/{dataset}/random'
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

    # generate the gradient map
    target_model_output_root = args.target_model_output_root
    save_target_model_output_gradient_root = f'{srresnet_gradient_save_folder}/{dataset}/targetmodel-sr'
    os.makedirs(save_target_model_output_gradient_root, exist_ok=True)

    faig_root = save_mask_faig_filters_folder
    save_faig_gradient_root = f'{srresnet_gradient_save_folder}/{dataset}/faig'
    os.makedirs(save_faig_gradient_root, exist_ok=True)

    ig_root = save_mask_ig_filters_folder
    save_ig_gradient_root = f'{srresnet_gradient_save_folder}/{dataset}/ig'
    os.makedirs(save_ig_gradient_root, exist_ok=True)

    abs_filter_change_root = save_mask_abs_filter_change_filters_folder
    save_abs_filter_change_gradient_root = f'{srresnet_gradient_save_folder}/{dataset}/abs_filter_change'
    os.makedirs(save_abs_filter_change_gradient_root, exist_ok=True)

    random_root = save_mask_random_filters_folder
    save_random_gradient_root = f'{srresnet_gradient_save_folder}/{dataset}/random'
    os.makedirs(save_random_gradient_root, exist_ok=True)

    print('Now we generate the gradient maps~ ')
    generate_gradient_maps_within_several_folders(
        net_type,
        target_model_output_root=target_model_output_root,
        save_target_model_output_gradient_root=save_target_model_output_gradient_root,
        faig_root=faig_root,
        save_faig_gradient_root=save_faig_gradient_root,
        ig_root=ig_root,
        save_ig_gradient_root=save_ig_gradient_root,
        abs_filter_change_root=abs_filter_change_root,
        save_abs_filter_change_gradient_root=save_abs_filter_change_gradient_root,
        random_root=random_root,
        save_random_gradient_root=save_random_gradient_root)

    print('Now we calculate the output difference between the target model and the constituted model~ ')
    print('===========================================================')
    # calculate the gradient loss
    # First calculate the loss of blurry inputs
    target_deblur_func_folder = os.path.join(save_target_model_output_gradient_root, blur_folder_name)
    os.makedirs(f'{srresnet_metric_save_folder}/{dataset}', exist_ok=True)

    save_faig_maskdeblurfilter_blur_txt = f'{srresnet_metric_save_folder}/{dataset}/faig_maskdeblurfilter_blur.txt'
    save_faig_maskdenoisefilter_blur_txt = f'{srresnet_metric_save_folder}/{dataset}/faig_maskdenoisefilter_blur.txt'

    save_ig_maskdeblurfilter_blur_txt = f'{srresnet_metric_save_folder}/{dataset}/ig_maskdeblurfilter_blur.txt'
    save_ig_maskdenoisefilter_blur_txt = f'{srresnet_metric_save_folder}/{dataset}/ig_maskdenoisefilter_blur.txt'

    save_abs_filter_change_maskdeblurfilter_blur_txt = f'{srresnet_metric_save_folder}/{dataset}/abs_filter_change_maskdeblurfilter_blur.txt'  # noqa E501
    save_abs_filter_change_maskdenoisefilter_blur_txt = f'{srresnet_metric_save_folder}/{dataset}/abs_filter_change_maskdenoisefilter_blur.txt'  # noqa E501

    save_random_maskdeblurfilter_blur_txt = f'{srresnet_metric_save_folder}/{dataset}/random_maskdeblurfilter_blur.txt'
    save_random_maskdenoisefilter_blur_txt = f'{srresnet_metric_save_folder}/{dataset}/random_maskdenoisefilter_blur.txt'  # noqa E501

    calc_blur_gradient_loss(
        net_type,
        target_deblur_func_folder,
        save_faig_gradient_root,
        save_faig_maskdeblurfilter_blur_txt,
        save_faig_maskdenoisefilter_blur_txt,
        save_ig_gradient_root,
        save_ig_maskdeblurfilter_blur_txt,
        save_ig_maskdenoisefilter_blur_txt,
        save_abs_filter_change_gradient_root,
        save_abs_filter_change_maskdeblurfilter_blur_txt,
        save_abs_filter_change_maskdenoisefilter_blur_txt,
        save_random_gradient_root,
        save_random_maskdeblurfilter_blur_txt,
        save_random_maskdenoisefilter_blur_txt,
        sub_func_folder_names=['maskdeblurfilter', 'maskdenoisefilter'],
        sub_input_folder_name=blur_folder_name)

    print('===========================================================')
    # Then calculate the loss of noisy inputs
    target_denoise_func_folder = os.path.join(save_target_model_output_gradient_root, noise_folder_name)

    save_faig_maskdeblurfilter_noise_txt = f'{srresnet_metric_save_folder}/{dataset}/faig_maskdeblurfilter_noise.txt'
    save_faig_maskdenoisefilter_noise_txt = f'{srresnet_metric_save_folder}/{dataset}/faig_maskdenoisefilter_noise.txt'

    save_ig_maskdeblurfilter_noise_txt = f'{srresnet_metric_save_folder}/{dataset}/ig_maskdeblurfilter_noise.txt'
    save_ig_maskdenoisefilter_noise_txt = f'{srresnet_metric_save_folder}/{dataset}/ig_maskdenoisefilter_noise.txt'

    save_abs_filter_change_maskdeblurfilter_noise_txt = f'{srresnet_metric_save_folder}/{dataset}/abs_filter_change_maskdeblurfilter_noise.txt'  # noqa E501
    save_abs_filter_change_maskdenoisefilter_noise_txt = f'{srresnet_metric_save_folder}/{dataset}/abs_filter_change_maskdenoisefilter_noise.txt'  # noqa E501

    save_random_maskdeblurfilter_noise_txt = f'{srresnet_metric_save_folder}/{dataset}/random_maskdeblurfilter_noise.txt'  # noqa E501
    save_random_maskdenoisefilter_noise_txt = f'{srresnet_metric_save_folder}/{dataset}/random_maskdenoisefilter_noise.txt'  # noqa E501

    calc_noise_gradient_loss(
        net_type,
        target_denoise_func_folder,
        save_faig_gradient_root,
        save_faig_maskdenoisefilter_noise_txt,
        save_faig_maskdeblurfilter_noise_txt,
        save_ig_gradient_root,
        save_ig_maskdenoisefilter_noise_txt,
        save_ig_maskdeblurfilter_noise_txt,
        save_abs_filter_change_gradient_root,
        save_abs_filter_change_maskdenoisefilter_noise_txt,
        save_abs_filter_change_maskdeblurfilter_noise_txt,
        save_random_gradient_root,
        save_random_maskdenoisefilter_noise_txt,
        save_random_maskdeblurfilter_noise_txt,
        sub_func_folder_names=['maskdeblurfilter', 'maskdenoisefilter'],
        sub_input_folder_name=noise_folder_name)


if __name__ == '__main__':
    main()
