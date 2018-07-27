import os
import argparse
from utils import create_dataset
from network import MobileNetv2_DeepLabv3
from config import Params
from utils import print_config


LOG = lambda x: print('\033[0;31;2m' + x + '\033[0m')


def main():
    # add argumentation
    parser = argparse.ArgumentParser(description='MobileNet_v2_DeepLab_v3 Pytorch Implementation')
    parser.add_argument('--dataset', default='cityscapes', choices=['cityscapes', 'other'],
                        help='Dataset used in training MobileNet v2+DeepLab v3')
    parser.add_argument('--root', default='./data/cityscapes', help='Path to your dataset')
    parser.add_argument('--epoch', default=None, help='Total number of training epoch')
    parser.add_argument('--lr', default=None, help='Base learning rate')
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
    if args.epoch is not None:
        params.num_epoch = args.epoch
    if args.lr is not None:
        params.base_lr = args.lr
    if args.pretrain is not None:
        params.pre_trained_from = args.pretrain
    if args.resume_from is not None:
        params.resume_from = args.resume_from

    LOG('Network parameters:')
    print_config(params)

    # create dataset and transformation
    LOG('Creating Dataset and Transformation......')
    datasets = create_dataset(params)
    LOG('Creation Succeed.\n')

    # create model
    LOG('Initializing MobileNet and DeepLab......')
    net = MobileNetv2_DeepLabv3(params, datasets)
    LOG('Model Built.\n')

    # let's start to train!
    net.Train()
    net.Test()

if __name__ == '__main__':
    main()