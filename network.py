from __future__ import print_function
import torch
import torch.nn as nn
import os
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import layers
from progressbar import bar

WARNING = lambda x: '\033[1;31;2mWARNING: ' + x + '\033[0m'

# create model
class MobileNetv2_DeepLabv3(nn.Module):
    """
    A Convolutional Neural Network with MobileNet v2 backbone and DeepLab v3 head
        used for Semantic Segmentation on Cityscapes dataset
    """

    """######################"""
    """# Model Construction #"""
    """######################"""

    def __init__(self, params, datasets):
        super(MobileNetv2_DeepLabv3, self).__init__()
        self.params = params
        self.datasets = datasets
        self.pb = bar()  # hand-made progressbar
        self.epoch = 0
        self.train_loss = []
        self.val_loss = []
        self.summary_writer = SummaryWriter(log_dir=self.params.summary_dir)

        # build network
        block = []

        # conv layer 1
        block.append(nn.Sequential(nn.Conv2d(3, self.params.c[0], 3, stride=self.params.s[0], padding=1, bias=False),
                                   nn.BatchNorm2d(self.params.c[0]),
                                   # nn.Dropout2d(self.params.dropout_prob, inplace=True),
                                   nn.ReLU6()))

        # conv layer 2-7
        for i in range(6):
            block.extend(layers.get_inverted_residual_block_arr(self.params.c[i], self.params.c[i+1],
                                                                t=self.params.t[i+1], s=self.params.s[i+1],
                                                                n=self.params.n[i+1]))

        # dilated conv layer 1-3
        rate = self.params.down_sample_rate // self.params.output_stride
        for i in range(3):
            block.append(layers.InvertedResidual(self.params.c[6], self.params.c[6],
                                                 t=self.params.t[6], s=1, dilation=rate*self.params.multi_grid[i]))

        # ASPP layer
        block.append(layers.ASPP_plus(self.params))

        # final conv layer
        block.append(nn.Conv2d(256, self.params.num_class, 1))

        self.network = nn.Sequential(*block).cuda()
        # print(self.network)

        # build loss
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=255)

        # optimizer
        self.opt = torch.optim.RMSprop(self.network.parameters(),
                                       lr=self.params.base_lr,
                                       momentum=self.params.momentum,
                                       weight_decay=self.params.weight_decay)

        # initialize
        self.initialize()

        # load data
        self.load_checkpoint()
        self.load_model()

    def initialize(self):
        """Initializes the model parameters"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    """######################"""
    """# Train and Validate #"""
    """######################"""

    def train_one_epoch(self):
        """
        Train network in one epoch
        """
        print('Training......')

        # set mode train
        self.network.train()

        # prepare data
        train_loss = 0
        train_loader = DataLoader(self.datasets['train'],
                                  batch_size=self.params.train_batch,
                                  shuffle=self.params.shuffle,
                                  num_workers=self.params.dataloader_workers)
        train_size = len(self.datasets['train'])
        if train_size % self.params.train_batch != 0:
            total_batch = train_size // self.params.train_batch + 1
        else:
            total_batch = train_size // self.params.train_batch

        # train through dataset
        for batch_idx, batch in enumerate(train_loader):
            self.pb.click(batch_idx, total_batch)
            image, label = batch['image'], batch['label']
            image_cuda, label_cuda = image.cuda(), label.cuda()
            out = self.network(image_cuda)
            loss = self.loss_fn(out, label_cuda)

            # optimize
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

            # accumulate
            train_loss += loss.item()

        self.pb.close()
        train_loss /= total_batch
        self.train_loss.append(train_loss)

        # add to summary
        self.summary_writer.add_scalar('train_loss', train_loss, self.epoch)

    def val_one_epoch(self):
        """
        Validate network in one epoch every m training epochs,
            m is defined in params.val_every
        """
        # TODO: add IoU compute function
        print('Validating:')

        # set mode eval
        self.network.eval()

        # prepare data
        val_loss = 0
        val_loader = DataLoader(self.datasets['val'],
                                batch_size=self.params.val_batch,
                                shuffle=self.params.shuffle,
                                num_workers=self.params.dataloader_workers)
        val_size = len(self.datasets['val'])
        if val_size % self.params.val_batch != 0:
            total_batch = val_size // self.params.val_batch + 1
        else:
            total_batch = val_size // self.params.val_batch

        # validate through dataset
        for batch_idx, batch in enumerate(val_loader):
            self.pb.click(batch_idx, total_batch)
            image, label = batch['image'], batch['label']
            image_cuda, label_cuda = image.cuda(), label.cuda()
            out = self.network(image_cuda)
            loss = self.loss_fn(out, label_cuda)

            val_loss += loss.item()

        self.pb.close()
        val_loss /= total_batch
        self.val_loss.append(val_loss)

        # softmax out
        # softmax = nn.Softmax(dim=1)
        # out = softmax(out)
        # out = torch.argmax(out, dim=1)

        # add to summary
        # TODO: add label visualization function and add visualized output into summary
        self.summary_writer.add_scalar('val_loss', val_loss, self.epoch)
        # self.summary_writer.add_image('epoch_%d_val_img' % self.epoch, image[0, ...], self.epoch)
        # self.summary_writer.add_image('epoch_%d_val_gt' % self.epoch, label[0, ...], self.epoch)
        # self.summary_writer.add_image('epoch_%d_val_seg' % self.epoch, out[0, ...], self.epoch)

    def Train(self):
        """
        Train network in n epochs, n is defined in params.num_epoch
        """
        for _ in range(self.params.num_epoch):
            self.epoch += 1
            print('-' * 20 + 'Epoch.' + str(self.epoch) + '-' * 20)

            # train one epoch
            self.train_one_epoch()

            # should display
            if self.epoch % self.params.display == 0:
                print('\tTrain loss: %.4f' % self.train_loss[-1])

            # should save
            if self.params.should_save:
                if self.epoch % self.params.save_every == 0:
                    self.save_checkpoint()

            # test every params.test_every epoch
            if self.params.should_val:
                if self.epoch % self.params.val_every == 0:
                    self.val_one_epoch()
                    print('\tVal loss: %.4f' % self.val_loss[-1])

            # adjust learning rate
            self.adjust_lr()

        # save the last network state
        self.save_checkpoint()

        # TODO: add train visualization

    def Test(self):
        """
        Test network on test set
        """

        # set mode eval
        self.network.eval()
        test_loader = DataLoader(self.datasets['test'],
                                 batch_size=self.params.test_batch,
                                 shuffle=False, num_workers=self.params.dataloader_workers)
        test_size = len(self.datasets['test'])
        if test_size % self.params.test_batch != 0:
            total_batch = test_size // self.params.test_batch + 1
        else:
            total_batch = test_size // self.params.test_batch

        for batch_idx, batch in enumerate(test_loader):
            self.pb.click(batch_idx, total_batch)
            image, label = batch
            image_cuda, label_cuda = image.cuda(), label.cuda()
            out = self.network(image_cuda)

            for i in range(self.params.test_batch):
                idx = batch_idx*self.params.test_batch+i
                self.summary_writer.add_image('test_img_%d' % idx, image[i, ...])
                self.summary_writer.add_image('test_out_%d' % idx, out[i, ...])

    def adjust_lr(self):
        """
        Adjust learning rate at each epoch
        """
        learning_rate = self.params.base_lr * (1 - float(self.epoch) / self.params.num_epoch) ** self.params.power
        for param_group in self.opt.param_groups:
            param_group['lr'] = learning_rate
        print('Change learning rate into %f' % (learning_rate))
        self.summary_writer.add_scalar('learning_rate', learning_rate, self.epoch)


    """##########################"""
    """# Model Save and Restore #"""
    """##########################"""

    def save_checkpoint(self):
        save_dict = {'epoch': self.epoch,
                     'state_dict': self.network.state_dict(),
                     'optimizer': self.opt.state_dict()}
        torch.save(save_dict, self.params.ckpt_dir+'Checkpoint_epoch_%d.pth.tar' % self.epoch)

    def load_checkpoint(self):
        if self.params.resume_from is not None and os.path.exists(self.params.resume_from):
            try:
                print('Loading Checkpoint at %s' % self.params.resume_from)
                ckpt = torch.load(self.params.resume_from)
                self.epoch = ckpt['epoch']
                self.network.load_state_dict(ckpt['state_dict'])
                self.opt.load_state_dict(ckpt['optimizer'])
                print('Checkpoint Loaded!')
                print('Current Epoch: %d' % self.epoch)
            except:
                print(WARNING('Cannot load checkpoint from %s. Skipping......' % self.params.resume_from))
        else:
            print(WARNING('Checkpoint do not exists. Skipping......'))

    def load_model(self):
        if self.params.pre_trained_from is not None and os.path.exists(self.params.pre_trained_from):
            try:
                print('Loading Pre-trained Model at %s' % self.params.pre_trained_from)
                pretrain = torch.load(self.params.pre_trained_from)
                self.network.load_state_dict(pretrain)
                print('Pre-trained Model Loaded!')
            except:
                print(WARNING('Cannot load pre-trained model. Start initializing......'))
        else:
            print(WARNING('Pre-trained model do not exits. Start initializing......'))


# """ TEST """
# if __name__ == '__main__':
#     params = CIFAR100_params()
#     params.dataset_root = '/home/ubuntu/cifar100'
#     net = MobileNetv2(params)
#     net.save_checkpoint()