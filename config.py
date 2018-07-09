from utils import create_train_dir


""" Dataset parameters """
class Params():
    # network structure parameters
    model = 'MobileNetv2_DeepLabv3'
    dataset = 'cityscapes'
    s = [2, 1, 2, 2, 2, 1, 1]  # stride of each conv stage
    t = [1, 1, 6, 6, 6, 6, 6]  # expansion factor t
    n = [1, 1, 2, 3, 4, 3, 3]  # number of repeat time
    c = [32, 16, 24, 32, 64, 96, 160]  # output channel of each conv stage
    output_stride = 16
    multi_grid = (1, 2, 4)
    aspp = (6, 12, 18)
    down_sample_rate = 32  # classic down sample rate

    # dataset parameters
    image_size = 512
    num_class = 20  # 20 classes for training
    dataset_root = '/path/to/your/dataset'
    dataloader_workers = 8
    shuffle = True
    train_batch = 5
    val_batch = 2
    test_batch = 5

    # train parameters
    num_epoch = 50
    base_lr = 0.001
    power = 0.9
    momentum = 0.9
    dropout_prob = 0.2
    weight_decay = 0.0005
    should_val = True
    val_every = 1
    display = 1  # show train result every display epoch

    # model restore parameters
    resume_from = None  # None for train from scratch
    pre_trained_from = None # None for train from scratch
    should_save = True
    save_every = 10

    def __init__(self):
        # create training dir
        self.summary_dir, self.ckpt_dir = create_train_dir(self)

# if __name__ == '__main__':
