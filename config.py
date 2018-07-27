from utils import create_train_dir


""" Dataset parameters """
class Params():
    def __init__(self):
        # network structure parameters
        self.model = 'MobileNetv2_DeepLabv3'
        self.dataset = 'cityscapes'
        self.s = [2, 1, 2, 2, 2, 1, 1]  # stride of each conv stage
        self.t = [1, 1, 6, 6, 6, 6, 6]  # expansion factor t
        self.n = [1, 1, 2, 3, 4, 3, 3]  # number of repeat time
        self.c = [32, 16, 24, 32, 64, 96, 160]  # output channel of each conv stage
        self.output_stride = 16
        self.multi_grid = (1, 2, 4)
        self.aspp = (6, 12, 18)
        self.down_sample_rate = 32  # classic down sample rate

        # dataset parameters
        self.rescale_size = 600
        self.image_size = 512
        self.num_class = 20  # 20 classes for training
        self.dataset_root = '/path/to/your/dataset'
        self.dataloader_workers = 12
        self.shuffle = True
        self.train_batch = 10
        self.val_batch = 2
        self.test_batch = 1

        # train parameters
        self.num_epoch = 150
        self.base_lr = 0.0002
        self.power = 0.9
        self.momentum = 0.9
        self.weight_decay = 0.0005
        self.should_val = True
        self.val_every = 2
        self.display = 1  # show train result every display epoch
        self.should_split = True  # should split training procedure into several parts
        self.split = 2  # number of split

        # model restore parameters
        self.resume_from = None  # None for train from scratch
        self.pre_trained_from = None  # None for train from scratch
        self.should_save = True
        self.save_every = 10

        # create training dir
        self.summary_dir, self.ckpt_dir = create_train_dir(self)

if __name__ == '__main__':
    aa = Params()
    print(aa)