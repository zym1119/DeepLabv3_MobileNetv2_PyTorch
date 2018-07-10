# DeepLabv3_MobileNetv2
This is a PyTorch implementation of MobileNet v2 network with DeepLab v3 structure used for semantic segmentation.

The backbone of MobileNetv2 comes from paper:
>[Inverted Residuals and Linear Bottlenecks: Mobile Networks for Classification, Detection and Segmentation ](https://arxiv.org/abs/1801.04381v3)

And the segment head of DeepLabv3 comes from paper:
>[Rethinking Atrous Convolution for Semantic Image Segmentation](https://arxiv.org/abs/1706.05587)

Please refer to these papers about details like Atrous Convolution, Inverted Residuals, Depthwise Convolution or ASPP if you have some confusion about these blocks.

# How to use?
First you need to install dependencies of this implementation.
This implementation is written under Python 3.5 with following libs:
>torch 0.4.0
torchvision 0.2.1
numpy 1.14.5
opencv-python 3.4.1.15
tensorflow 1.8.0 (necessary for tensorboardX)
tensorboardX 1.2

use `sudo pip install lib` to install them

Then, prepare cityscapes dataset or your own dataset.
Currently, cityscapes is the only supported dataset without any modification.

Cityscapes dataset should have the following hierachy:
```
dataset_root
|   trainImages.txt
|   trainLabels.txt
|   valImages.txt
|   valLabels.txt 
|
└───gtFine(Label Folder)
|   └───train(train set)
|   |   └───aachen(city)
|   |   └───bochum
|   |   └───...
|   |   
|   └───test(test set)
|   └───val(val set)
|
└───leftImg8bit(Image Folder)
    └───train
    └───test
    └───val
```

Third, modify `config.py` to fit your own training policy or configuration

At last, run `python main.py --root /your/path/to/dataset/` or just run `python main.py`

After training, tensorboard is also available to observe training procedure using `tensorboard --logdir=./exp_dir/summaries`

If you have some question, please leave an issue.
