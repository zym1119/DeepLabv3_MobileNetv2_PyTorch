from __future__ import print_function
import os
import argparse
from config import Params, CIFAR100_params, CIFAR10_params
from utils import create_dataset
from mobilenetv2 import MobileNetv2


def main():
    parser = argparse.ArgumentParser(description='MobileNet_V2 Pytorch Implementation')
    parser.add_argument('--dataset', default='cifar10', choices=['imagenet', 'cifar10', 'cifar100', 'other'],
                        help='Dataset used in training MobileNet V2')
    parser.add_argument('--root', default='./data/cifar10', help='Path to your dataset')

    args = parser.parse_args()

    # parse args
    if args.dataset == 'cifar10':
        params = CIFAR10_params()
    elif args.dataset == 'cifar100':
        params = CIFAR100_params()
    else:
        params = Params()
    params.dataset_root = args.root

    if not os.path.exists(args.root):
        print('ERROR: Root %s not exists!' % args.root)
        exit(1)

    """ TEST CODE """
    # params = CIFAR100_params
    # params.dataset_root = '/home/ubuntu/cifar100'

    # create model
    print('\nInitializing MobileNet......')
    net = MobileNetv2(params)
    print('Initialization Done.\n')

    # create dataset and transformation
    print('Loading Data......')
    dataset = create_dataset(params)
    print('Data Loaded.\n')

    # let's start to train!
    net.train_n_epoch(dataset)

if __name__ == '__main__':
    main()