import torch
from config import Params
from network import MobileNetv2_DeepLabv3, LOG
from utils import create_dataset


source_weight = '/home/ubuntu/Downloads/model_best.pth.tar'

weight = torch.load(source_weight, map_location='cuda:0')
state_dict = weight['state_dict']

params = Params()
params.dataset_root = '/media/ubuntu/disk/cityscapes'
datasets = create_dataset(params)
LOG('Creation Succeed.\n')

# create model
LOG('Initializing MobileNet and DeepLab......')
net = MobileNetv2_DeepLabv3(params, datasets)

index = 0
my_net_keys = list(net.state_dict().keys())
my_net_weights = list(net.state_dict().values())
for w in state_dict.values():
    if my_net_weights[index].shape == w.shape:
        net.state_dict()[my_net_keys[index]] = w
        print('Store weight in %s layer' % my_net_keys[index])
    index += 1
torch.save(net.state_dict(), './ImageNet_pretrain.pth')