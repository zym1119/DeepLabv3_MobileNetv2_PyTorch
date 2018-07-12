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
Don't worry about txt files if you don't have them, this program can generate unexist txt files automatically.

Third, modify `config.py` to fit your own training policy or configuration

At last, run `python main.py --root /your/path/to/dataset/` or just run `python main.py`

After training, tensorboard is also available to observe training procedure using `tensorboard --logdir=./exp_dir/summaries`

# Tips
I have changed a little from origin MobileNetv2 and DeepLabv3 network, here are the changes:
```
1. The multi-grid blocks have the same structure with the 7-th layer in MobileNetv2 while 
the rest layers of MobileNetv2 are discarded.
2. The lr decay is determined by epoch not iterations as in DeepLab and the input image 
is randomly cropped by 512 instead of 513 in DeepLab.
3. During training, a input image is first resized so that the shorter side is 600 pixel, then cropped into 512 pixels square and sent into network.
```

If you have some question, please leave an issue.

# TO-DO
1. add cityscapes visualization tools
2. fine-tune training policy

# Logs
| 7.11 | fix bugs in network.Test(), add cityscapes output visualization function |
|------|-------------------------------------------------------------------|
| 7.12 | fix bugs in network.plot_curve(), add checkpoint split to avoid out of memory, add save loss in network.save_checkpoint() |
