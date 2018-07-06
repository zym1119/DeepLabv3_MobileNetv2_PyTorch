from __future__ import print_function
import torch
import torch.nn as nn
import os
import numpy as np
from tensorboardX import SummaryWriter
import layers
from progressbar import bar
from config import CIFAR100_params

WARNING = lambda x: '\033[1;31;2m WARNING: ' + x + '\033[0m'

# create model
class MobileNetv2(nn.Module):
    def __init__(self, params):
        super(MobileNetv2, self).__init__()
        self.params = params
        self.pb = bar()  # hand-made progressbar
        self.epoch = 0
        self.test_epoch = 0
        self.train_loss = 0
        self.test_loss = 0
        self.train_acc = 0
        self.test_acc = 0
        self.summary_writer = SummaryWriter(log_dir=self.params.summary_dir)

        # build network
        block = []

        # conv layer 1
        block.append(nn.Sequential(nn.Conv2d(3, self.params.c[0], 3, stride=1, padding=1, bias=False),
                                   nn.BatchNorm2d(self.params.c[0]),
                                   nn.Dropout2d(self.params.dropout_prob, inplace=True),
                                   nn.ReLU6()))

        # conv layer 2-8
        for i in range(7):
            block.extend(layers.get_inverted_residual_block_arr(self.params.c[i], self.params.c[i+1],
                                                                t=self.params.t[i+1], s=self.params.s[i+1],
                                                                n=self.params.n[i+1]))

        # conv layer 9
        block.append(nn.Sequential(nn.Conv2d(self.params.c[-2], self.params.c[-1], 1, bias=False),
                                   nn.BatchNorm2d(self.params.c[-1]),
                                   nn.ReLU6()))

        # pool and fc
        block.append(nn.Sequential(nn.AvgPool2d(self.params.image_size//self.params.down_sample_rate),
                                   nn.Dropout2d(self.params.dropout_prob, inplace=True),
                                   nn.Conv2d(self.params.c[-1], self.params.num_class, 1, bias=True)))

        self.network = nn.Sequential(*block).cuda()
        # print(self.network)

        # build loss
        self.loss_fn = nn.CrossEntropyLoss().cuda()

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

    def adjust_lr(self):
        """
        Adjust learning rate at each epoch
        """
        learning_rate = self.params.base_lr * self.params.lr_decay**self.epoch
        for param_group in self.opt.param_groups:
            param_group['lr'] = learning_rate
        print('Change learning rate into %f' % (learning_rate))
        self.summary_writer.add_scalar('learning_rate', learning_rate, self.epoch)

    def train_one_epoch(self, dataset):
        print('Training:')

        # set train mode
        self.network.train()

        # prepare data
        self.train_loss, self.train_acc = 0, 0
        train_loader = torch.utils.data.DataLoader(dataset['train'],
                                                   batch_size=self.params.train_batch,
                                                   shuffle=self.params.shuffle,
                                                   num_workers=self.params.dataloader_workers)
        train_size = len(dataset['train'])
        if train_size % self.params.train_batch != 0:
            total_batch = train_size//self.params.train_batch+1
        else:
            total_batch = train_size//self.params.train_batch

        # train through dataset
        for batch_idx, batch in enumerate(train_loader):
            self.pb.click(batch_idx, total_batch)
            image, label = batch
            image_cuda, label_cuda  = image.cuda(), label.cuda()
            out = self.network(image_cuda).squeeze_()
            loss = self.loss_fn(out, label_cuda)

            # optimize
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

            # accuracy
            acc = np.mean(np.argmax(out.cpu().detach().numpy(), axis=1) == label.numpy())

            self.train_loss += loss.item()
            self.train_acc += acc

        self.pb.close()
        self.train_loss /= total_batch
        self.train_acc /= total_batch

        # add to summary
        self.summary_writer.add_scalar('train_loss', self.train_loss, self.epoch)
        self.summary_writer.add_scalar('train_acc', self.train_acc, self.epoch)

    def test_one_epoch(self, dataset):
        print('Testing:')

        # set mode test
        self.network.eval()

        # prepare data
        val_loader = torch.utils.data.DataLoader(dataset['val'],
                                                 batch_size=self.params.test_batch,
                                                 shuffle=self.params.shuffle,
                                                 num_workers=self.params.dataloader_workers)
        test_size = len(dataset['val'])
        if test_size % self.params.test_batch != 0:
            total_batch = test_size // self.params.test_batch + 1
        else:
            total_batch = test_size // self.params.test_batch

        # test through dataset
        for batch_idx, batch in enumerate(val_loader):
            self.pb.click(batch_idx, total_batch)
            image, label = batch
            image_cuda, label_cuda = image.cuda(), label.cuda()
            out = self.network(image_cuda).squeeze_()
            loss = self.loss_fn(out, label_cuda)
            acc = np.mean(np.argmax(out.cpu().detach().numpy(), axis=1) == label.numpy())

            self.test_loss += loss.item()
            self.test_acc += acc

        self.pb.close()
        self.test_loss /= total_batch
        self.test_acc /= total_batch

        # add to summary
        self.summary_writer.add_scalar('test_loss', self.test_loss, self.epoch)
        self.summary_writer.add_scalar('test_acc', self.test_acc, self.epoch)

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
                print(WARNING('Cannot load checkpoint from %s. Continue optimizing......' % self.params.resume_from))
        else:
            print(WARNING('Checkpoint not exists. Continue optimizing......'))

    def load_model(self):
        if self.params.pre_trained_from is not None and os.path.exists(self.params.pre_trained_from):
            try:
                print('Loading Pre-trained Model at %s' % self.params.pre_trained_from)
                pretrain = torch.load(self.params.pre_trained_from)
                self.network.load_state_dict(pretrain)
                print('Pre-trained Model Loaded!')
            except:
                print(WARNING('Cannot load pre-trained model. Continue optimizing......'))
        else:
            print(WARNING('Model not exits. Continue optimizing......'))

    def train_n_epoch(self, dataset):
        """
        Train network for n epoch, n is defined in params
        :param dataset: dataset
        """
        for epoch in range(self.params.num_epoch):
            self.epoch += 1
            print('-' * 20 + 'Epoch.' + str(self.epoch) + '-' * 20)
            self.train_one_epoch(dataset)
            print('\tTrain acc: %.2f, Train loss: %.4f' % (self.train_acc*100, self.train_loss))
            # should save
            if self.epoch % self.params.save_every == 0:
                self.save_checkpoint()

            # test every params.test_every epoch
            if self.params.should_test:
                if self.test_epoch % self.params.test_every == 0:
                    self.test_one_epoch(dataset)
                    print('\tTest acc: %.2f, Test loss: %.4f' % (self.test_acc*100, self.test_loss))

            # save model
            if self.params.should_save:
                if self.epoch % self.params.save_every == 0:
                    self.save_checkpoint()

            self.adjust_lr()


""" TEST """
if __name__ == '__main__':
    params = CIFAR100_params()
    params.dataset_root = '/home/ubuntu/cifar100'
    net = MobileNetv2(params)
    net.save_checkpoint()