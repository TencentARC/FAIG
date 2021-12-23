import glob
import numpy as np
import os


def calc_noise_gradient_loss(net_type, target_noise_folder, faig_folder, save_faig_maskdenoisefilter_noise_loss_txt,
                             save_faig_maskdeblurfilter_noise_loss_txt, ig_folder,
                             save_ig_maskdenoisefilter_noise_loss_txt, save_ig_maskdeblurfilter_noise_loss_txt,
                             abs_filter_change_folder, save_abs_filter_change_maskdenoisefilter_noise_loss_txt,
                             save_abs_filter_change_maskdeblurfilter_noise_loss_txt, random_folder,
                             save_random_maskdenoisefilter_noise_loss_txt, save_random_maskdeblurfilter_noise_loss_txt,
                             sub_func_folder_names, sub_input_folder_name):
    """ Quantity the discovered filters' contribution to the deblur function by measuring
        output difference of the target model and the substituted model. The output difference
        is calculated on image gradients of their gray counterpart.

    Args:
        net_type (str): network type. Default: srcnn_style or srresnet
        target_noise_folder (str): folder path that contains the gradient map of target model's output
            towards blurry input.
        faig_folder (str): folder path that contains the gradient map of substituted-faig-discovered model's
            output towards blurry input.
        save_faig_maskdenoisefilter_noise_loss_txt (str): txt path that records the output different of
            target model and substituted-faig-discovered (noise) model.
        save_faig_maskdeblurfilter_noise_loss_txt (str): txt path that records the output different of
            target model and substituted-faig-discovered (blur) model.
        ig_folder (str): folder path that contains the gradient map of substituted-ig-discovered model's
            output towards blurry input.
        save_ig_maskdenoisefilter_noise_loss_txt (str): txt path that records the output different of
            target model and substituted-ig-discovered (noise) model.
        save_ig_maskdeblurfilter_noise_loss_txt (str): txt path that records the output different of
            target model and substituted-ig-discovered (blur) model.
        abs_filter_change_folder (str): folder path that contains the gradient map of
            substituted-abs_filter_change-discovered model's output towards blurry input.
        save_abs_filter_change_maskdenoisefilter_noise_loss_txt (str): txt path that records the output different of
            target model and substituted-abs_filter_change-discovered (noise) model.
        save_abs_filter_change_maskdeblurfilter_noise_loss_txt (str): txt path that records the output different of
            target model and substituted-abs_filter_change-discovered (blur) model.
        random_folder (str): folder path that contains the gradient map of substituted-random-discovered model's
            output towards blurry input.
        save_random_maskdenoisefilter_noise_loss_txt (str): txt path that records the output different of
            target model and substituted-random-discovered (noise) model.
        save_random_maskdeblurfilter_noise_loss_txt (str): txt path that records the output different of
            target model and substituted-random-discovered (blur) model.
        sub_func_folder_names (list): Default: ['maskdeblurfilter', 'maskdenoisefilter']
        sub_input_folder_name (str): 'LRbicx2_noise0.1'

    """

    denoise_func_imglist = list(sorted(glob.glob(os.path.join(target_noise_folder, '*.npy'))))

    faig_maskdenoisefilter_noise_loss = []
    faig_maskdeblurfilter_noise_loss = []

    ig_maskdenoisefilter_noise_loss = []
    ig_maskdeblurfilter_noise_loss = []

    abs_filter_change_maskdenoisefilter_noise_loss = []
    abs_filter_change_maskdeblurfilter_noise_loss = []

    random_maskdenoisefilter_noise_loss = []
    random_maskdeblurfilter_noise_loss = []

    if net_type == 'srcnn_style':
        total_neuron_nums = 156224
    elif net_type == 'srresnet':
        total_neuron_nums = 151936

    for proportion in [1, 3, 5, 10]:
        # for proportion in range(0, 101):
        selected_num_neurons = int(total_neuron_nums * proportion / 100)
        neuron_folder = f'{selected_num_neurons}kernels'

        faig_neuron_folder_path = os.path.join(faig_folder, neuron_folder)
        ig_neuron_folder_path = os.path.join(ig_folder, neuron_folder)
        abs_filter_change_neuron_folder_path = os.path.join(abs_filter_change_folder, neuron_folder)
        random_neuron_folder_path = os.path.join(random_folder, neuron_folder)

        for idx, sub_folder in enumerate(sub_func_folder_names):
            faig_neuron_sub_folder_path = os.path.join(faig_neuron_folder_path, sub_folder)
            ig_neuron_sub_folder_path = os.path.join(ig_neuron_folder_path, sub_folder)
            abs_filter_change_neuron_sub_folder_path = os.path.join(abs_filter_change_neuron_folder_path, sub_folder)
            random_neuron_sub_folder_path = os.path.join(random_neuron_folder_path, sub_folder)

            faig_imglist = list(
                sorted(glob.glob(os.path.join(faig_neuron_sub_folder_path, sub_input_folder_name, '*.npy'))))

            ig_imglist = list(
                sorted(glob.glob(os.path.join(ig_neuron_sub_folder_path, sub_input_folder_name, '*.npy'))))

            abs_filter_change_imglist = list(
                sorted(
                    glob.glob(os.path.join(abs_filter_change_neuron_sub_folder_path, sub_input_folder_name, '*.npy'))))

            random_imglist = list(
                sorted(glob.glob(os.path.join(random_neuron_sub_folder_path, sub_input_folder_name, '*.npy'))))
            faig_gradient_loss = 0.0
            ig_gradient_loss = 0.0
            abs_filter_change_gradient_loss = 0.0
            random_gradient_loss = 0.0

            for img_idx, img_path in enumerate(denoise_func_imglist):
                refer_img_path = img_path
                faig_img_path = faig_imglist[img_idx]
                ig_img_path = ig_imglist[img_idx]
                abs_filter_change_img_path = abs_filter_change_imglist[img_idx]
                random_img_path = random_imglist[img_idx]

                refer_gradient = np.load(refer_img_path)
                faig_gradient = np.load(faig_img_path)
                ig_gradient = np.load(ig_img_path)
                abs_filter_change_gradient = np.load(abs_filter_change_img_path)
                random_gradient = np.load(random_img_path)

                # for better visualization, we multiple the results with 1000
                faig_gradient_loss += np.mean((faig_gradient - refer_gradient)**2) * 1000
                ig_gradient_loss += np.mean((ig_gradient - refer_gradient)**2) * 1000
                abs_filter_change_gradient_loss += np.mean((abs_filter_change_gradient - refer_gradient)**2) * 1000
                random_gradient_loss += np.mean((random_gradient - refer_gradient)**2) * 1000

            faig_gradient_loss /= len(denoise_func_imglist)
            ig_gradient_loss /= len(denoise_func_imglist)
            abs_filter_change_gradient_loss /= len(denoise_func_imglist)
            random_gradient_loss /= len(denoise_func_imglist)

            if idx == 0:
                print('Calculate the effectiveness of masking discovered deblur filters for denoise function. '
                      'Lower value is better!')
                print(f'faig:{round(faig_gradient_loss, 4)}\t ig:{round(ig_gradient_loss, 4)}\t '
                      f'abs_filter_change:{round(abs_filter_change_gradient_loss, 4)}\t '
                      f'random:{round(random_gradient_loss, 4)}')

                faig_maskdenoisefilter_noise_loss.append(faig_gradient_loss)
                ig_maskdenoisefilter_noise_loss.append(ig_gradient_loss)
                abs_filter_change_maskdenoisefilter_noise_loss.append(abs_filter_change_gradient_loss)
                random_maskdenoisefilter_noise_loss.append(random_gradient_loss)

            else:
                print('Calculate the effectiveness of masking discovered denoise filters for denoise function. '
                      'Higher value is better!')
                print(f'faig:{round(faig_gradient_loss, 4)}\t ig:{round(ig_gradient_loss, 4)}\t '
                      f'abs_filter_change:{round(abs_filter_change_gradient_loss, 4)}\t '
                      f'random:{round(random_gradient_loss, 4)}')

                faig_maskdeblurfilter_noise_loss.append(faig_gradient_loss)
                ig_maskdeblurfilter_noise_loss.append(ig_gradient_loss)
                abs_filter_change_maskdeblurfilter_noise_loss.append(abs_filter_change_gradient_loss)
                random_maskdeblurfilter_noise_loss.append(random_gradient_loss)

    faig_maskdeblurfilter_noise_loss = np.array(faig_maskdeblurfilter_noise_loss)
    faig_maskdenoisefilter_noise_loss = np.array(faig_maskdenoisefilter_noise_loss)

    ig_maskdeblurfilter_noise_loss = np.array(ig_maskdeblurfilter_noise_loss)
    ig_maskdenoisefilter_noise_loss = np.array(ig_maskdenoisefilter_noise_loss)

    abs_filter_change_maskdeblurfilter_noise_loss = np.array(abs_filter_change_maskdeblurfilter_noise_loss)
    abs_filter_change_maskdenoisefilter_noise_loss = np.array(abs_filter_change_maskdenoisefilter_noise_loss)

    random_maskdeblurfilter_noise_loss = np.array(random_maskdeblurfilter_noise_loss)
    random_maskdenoisefilter_noise_loss = np.array(random_maskdenoisefilter_noise_loss)

    # write the result to txt
    np.savetxt(save_faig_maskdeblurfilter_noise_loss_txt, faig_maskdeblurfilter_noise_loss, delimiter=',', fmt='%.6f')
    np.savetxt(save_faig_maskdenoisefilter_noise_loss_txt, faig_maskdenoisefilter_noise_loss, delimiter=',', fmt='%.6f')

    np.savetxt(save_ig_maskdeblurfilter_noise_loss_txt, ig_maskdeblurfilter_noise_loss, delimiter=',', fmt='%.6f')
    np.savetxt(save_ig_maskdenoisefilter_noise_loss_txt, ig_maskdenoisefilter_noise_loss, delimiter=',', fmt='%.6f')

    np.savetxt(
        save_abs_filter_change_maskdeblurfilter_noise_loss_txt,
        abs_filter_change_maskdeblurfilter_noise_loss,
        delimiter=',',
        fmt='%.6f')
    np.savetxt(
        save_abs_filter_change_maskdenoisefilter_noise_loss_txt,
        abs_filter_change_maskdenoisefilter_noise_loss,
        delimiter=',',
        fmt='%.6f')

    np.savetxt(
        save_random_maskdeblurfilter_noise_loss_txt, random_maskdeblurfilter_noise_loss, delimiter=',', fmt='%.6f')
    np.savetxt(
        save_random_maskdenoisefilter_noise_loss_txt, random_maskdenoisefilter_noise_loss, delimiter=',', fmt='%.6f')
