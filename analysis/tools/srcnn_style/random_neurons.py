import argparse
import numpy as np
import os
import random


def random_neuron_index(args):
    """ Sort the filters randomly.
    """
    total_neuron_nums = args.total_neuron_nums
    record_filters_folder = args.record_filters_folder
    os.makedirs(record_filters_folder, exist_ok=True)

    save_noisy_filter_txt = os.path.join(record_filters_folder, 'noise_index.txt')
    random_noisy_location = list(range(0, total_neuron_nums))
    random.shuffle(random_noisy_location)
    np.savetxt(save_noisy_filter_txt, random_noisy_location, delimiter=',', fmt='%d')

    save_blurry_filter_txt = os.path.join(record_filters_folder, 'blur_index.txt')
    random_blurry_location = list(range(0, total_neuron_nums))
    random.shuffle(random_blurry_location)
    np.savetxt(save_blurry_filter_txt, random_blurry_location, delimiter=',', fmt='%d')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--total_neuron_nums', type=int, default=156224, help='the number of filters in srcnn')
    parser.add_argument(
        '--record_filters_folder',
        type=str,
        default='results/Interpret/neuron-search/srcnn_style/Set14/random',
        help='folder that saves the sorted location index of discovered filters')
    args = parser.parse_args()
    random_neuron_index(args)
