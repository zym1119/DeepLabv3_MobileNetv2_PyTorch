import os
import argparse
from utils import create_dataset
from network import MobileNetv2_DeepLabv3
from config import Params
from utils import print_config

def main():
    # add argumentation
    parser = argparse.ArgumentParser(description='MobileNet_v2_DeepLab_v3 Pytorch Implementation')
    parser.add_argument('--dataset', default='cityscapes', choices=['cityscapes', 'other'],
                        help='Dataset used in training MobileNet v2+DeepLab v3')
    parser.add_argument('--root', default='./data/cityscapes', help='Path to your dataset')
    parser.add_argument('--epoch', default=50, help='Total number of training epoch')
    parser.add_argument('--lr', default=0.00025, help='Base learning rate')
    parser.add_argument('--pretrain', default=None, help='Path to a pre-trained backbone model')
    parser.add_argument('--resume_from', default=None, help='Path to a checkpoint to resume model')

    args = parser.parse_args()
    params = Params()

    # parse args
    if not os.path.exists(args.root):
        if params.dataset_root is None:
            raise ValueError('ERROR: Root %s not exists!' % args.root)
    else:
        params.dataset_root = args.root
    params.num_epoch = args.epoch
    params.base_lr = args.lr
    params.pre_trained_from = args.pretrain
    params.resume_from = args.resume_from

    print('Network parameters:')
    print_config(params)

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