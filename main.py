from __future__ import print_function
import os
import argparse
from utils import create_dataset
from network import MobileNetv2_DeepLabv3
from config import Params


def main():
    # parser = argparse.ArgumentParser(description='MobileNet_v2_DeepLab_v3 Pytorch Implementation')
    # parser.add_argument('--dataset', default='cityscapes', choices=['cityscapes', 'other'],
    #                     help='Dataset used in training MobileNet v2+DeepLab v3')
    # parser.add_argument('--root', default='./data/cityscapes', help='Path to your dataset')
    #
    # args = parser.parse_args()

    # parse args
    # if not os.path.exists(args.root):
    #     print('ERROR: Root %s not exists!' % args.root)
    #     exit(1)

    params = Params()
    # params.dataset_root = args.root
    params.dataset_root = '/media/ubuntu/disk/cityscapes'
    """ TEST CODE """
    # create dataset and transformation
    print('Creating Dataset and Transformation......')
    datasets = create_dataset(params)
    print('Creation Succeed.\n')

    # create model
    print('Initializing MobileNet and DeepLab......')
    net = MobileNetv2_DeepLabv3(params, datasets)
    print('Initialization Succeed.\n')

    # let's start to train!
    net.Train()

if __name__ == '__main__':
    main()