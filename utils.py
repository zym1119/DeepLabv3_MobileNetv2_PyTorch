import numpy as np
import os
import cv2
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


def calc_dataset_stats(dataset, axis=0, ep=1e-7):
    return (np.mean(dataset, axis=axis) / 255.0).tolist(), (np.std(dataset + ep, axis=axis) / 255.0).tolist()


def create_train_dir(params):
    """
    Create folder used in training, folder hierarchy:
    current folder--exp_folder
                   |
                   --summaries
                   --checkpoints
    """
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
    """
    Create datasets for training, testing and validating
    :return datasets: a python dictionary includes three datasets
                        datasets[
    """
    phase = ['train', 'val', 'test']
    # if params.dataset_root is not None and not os.path.exists(params.dataset_root):
    #     raise ValueError('Dataset not exists!')

    transform = {'train': transforms.Compose([Rescale(params.rescale_size),
                                              RandomCrop(params.image_size),
                                              RandomHorizontalFlip(),
                                              ToTensor()
                                              ]),
                 'val'  : transforms.Compose([Rescale(params.image_size),
                                              ToTensor()
                                              ]),
                 'test' : transforms.Compose([Rescale(params.image_size),
                                              ToTensor()
                                              ])}

    # file_dir = {p: os.path.join(params.dataset_root, p) for p in phase}

    # datasets = {Cityscapes(file_dir[p], mode=p, transforms=transform[p]) for p in phase}
    datasets = {p: Cityscapes(params.dataset_root, mode=p, transforms=transform[p]) for p in phase}

    return datasets


class Cityscapes(Dataset):
    def __init__(self, dataset_dir, mode='train', transforms=None):
        """
        Create Dataset subclass on cityscapes dataset
        :param dataset_dir: the path to dataset root, eg. '/media/ubuntu/disk/cityscapes'
        :param mode: phase, 'train', 'test' or 'eval'
        :param transforms: transformation
        """
        self.dataset = dataset_dir
        self.transforms = transforms
        require_file = ['trainImages.txt', 'valImages.txt', 'trainLabels.txt', 'valLabels.txt']

        # check requirement
        if mode not in ['train', 'test', 'val']:
            raise ValueError('Unsupported mode %s' % mode)

        if not os.path.exists(self.dataset):
            raise ValueError('Dataset not exists!')

        for file in require_file:
            if file not in os.listdir(self.dataset):
                raise ValueError('Cannot find file %s in dataset root folder!' % file)

        # create image and label list
        self.image_list = []
        self.label_list = []
        if mode == 'train':
            for line in open(os.path.join(self.dataset, 'trainImages.txt')):
                self.image_list.append(line.strip())
            for line in open(os.path.join(self.dataset, 'trainLabels.txt')):
                self.label_list.append(line.strip())
        elif mode == 'val':
            for line in open(os.path.join(self.dataset, 'valImages.txt')):
                self.image_list.append(line.strip())
            for line in open(os.path.join(self.dataset, 'valLabels.txt')):
                self.label_list.append(line.strip())
        else:
            test_dir = os.path.join(self.dataset, 'leftImg8bit/val')
            cities = os.listdir(test_dir)
            for city in cities:
                self.image_list.extend(os.listdir(os.path.join(test_dir, city)))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        """
        Overrides default method
        tips: 3 channels of label image are the same
        """
        image = cv2.imread(os.path.join(self.dataset, self.image_list[index]))
        label = cv2.imread(os.path.join(self.dataset, self.label_list[index]))

        sample = {'image': image, 'label': label[:, :, 0]}  # label.size (1024, 2048, 3)

        if self.transforms:
            sample = self.transforms(sample)

        return sample


class Rescale(object):
    """
    Rescale the image in a sample to a given size.
    :param output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = cv2.resize(image, (new_w, new_h))
        label = cv2.resize(label, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        return {'image': img, 'label': label}


class ToTensor(object):
    """
    Convert ndarrays in sample to Tensors with normalization.
    """
    def __init__(self, output_stride=16):
        self.output_stride = output_stride

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1)).astype(np.float32)

        # reset label shape and set 255 into -1
        w, h = label.shape[0]//self.output_stride, label.shape[1]//self.output_stride
        label = cv2.resize(label, (h, w), interpolation=cv2.INTER_NEAREST).astype(np.int64)
        # label[label == 255] = 19

        # min-max normalize image
        immax = np.max(image)
        immin = np.min(image)
        image = (image-immin)/(immax-immin)

        return {'image': torch.from_numpy(image),
                'label': torch.from_numpy(label)}


class RandomHorizontalFlip(object):
    """
    Random flip image and label horizontally
    """
    def __call__(self, sample, p=0.5):
        image, label = sample['image'], sample['label']
        if np.random.uniform(0, 1) < p:
            image = cv2.flip(image, 1)
            label = cv2.flip(label, 1)

        return {'image': image, 'label': label}


class RandomCrop(object):
    """
    Crop randomly the image in a sample.

    :param output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h, left: left + new_w, :]

        label = label[top: top + new_h, left: left + new_w]

        return {'image': image, 'label': label}


def print_config(params):
    for name, value in sorted(vars(params).items()):
        print('\t%-20s:%s' % (name, str(value)))
    print('')


if __name__ == '__main__':
    dir = '/media/ubuntu/disk/cityscapes'
    dataset = Cityscapes(dir)
    loader = DataLoader(dataset,
                        batch_size=10,
                        shuffle=True,
                        num_workers=8)
    for idx, batch in enumerate(loader):
        img = batch['image']
        lb = batch['label']
        print(idx, img.shape)

    # tips: the last batch may not be as big as batch_size

