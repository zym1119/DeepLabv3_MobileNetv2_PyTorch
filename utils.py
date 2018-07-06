import numpy as np
import os
import torchvision
from torchvision import transforms


def calc_dataset_stats(dataset, axis=0, ep=1e-7):
    return (np.mean(dataset, axis=axis) / 255.0).tolist(), (np.std(dataset + ep, axis=axis) / 255.0).tolist()


def create_train_dir(params):
    experiment = params.model + '_' + params.dataset
    exp_dir = os.path.join(os.getcwd(), experiment)
    summary_dir = os.path.join(exp_dir, 'summaries/')
    checkpoint_dir = os.path.join(exp_dir, 'checkpoints/')

    dir = [exp_dir, summary_dir, checkpoint_dir]
    for dir_ in dir:
        if not os.path.exists(dir_):
            os.mkdir(dir_)

    return summary_dir, checkpoint_dir


def create_dataset(params):
    phase = ['train', 'val']
    if params.dataset_root is not None and not os.path.exists(params.dataset_root):
        raise ValueError('Dataset not exists!')

    if params.dataset == 'cifar100':
        mean, std = calc_dataset_stats(torchvision.datasets.CIFAR100(root=params.dataset_root, train=True, download=True).train_data,
                                       axis=(0, 1, 2))
    elif params.dataset == 'cifar10':
        mean, std = calc_dataset_stats(torchvision.datasets.CIFAR10(root=params.dataset_root, train=True, download=True).train_data,
                                       axis=(0, 1, 2))
    else:
        print("WARNING: No mean std given! Use default instead")
        # Or write your mean & std here~
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]

    transform = {'train': transforms.Compose([transforms.Resize(params.image_size),
                                              transforms.RandomCrop(params.cropped_size,
                                                                    padding=4),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ColorJitter(0.3, 0.3, 0.3),
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean, std)
                                              ]),
                 'val'  : transforms.Compose([transforms.ToTensor(),
                                              transforms.Normalize(mean, std)
                                              ])}
    if params.dataset == 'cifar10':
        dataset = {x: torchvision.datasets.CIFAR10(root=params.dataset_root, train= x == 'train',
                                                   transform=transform[x], download=True) for x in phase}
    elif params.dataset == 'cifar100':
        dataset = {x: torchvision.datasets.CIFAR100(root=params.dataset_root, train=x == 'train',
                                                    transform=transform[x], download=True) for x in phase}
    else:
        print('ERROR: No proper dataset defined, please define your own dataset!')
        exit(1)

    return dataset
