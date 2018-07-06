from utils import create_train_dir


""" Dataset parameters """
class Params():
    # network structure parameters
    model = 'MobileNet_V2'
    dataset = 'imagenet'
    t = [1, 1, 6, 6, 6, 6, 6, 6]  # expansion factor t
    s = [1, 1, 1, 2, 2, 1, 2, 1, 1]  # stride of each conv stage
    n = [1, 1, 2, 3, 4, 3, 3, 1, 1]  # number of repeat time
    c = [32, 16, 24, 32, 64, 96, 160, 320, 1280]  # output channel of each conv stage
    down_sample_rate = 32  # product of strides above

    # dataset parameters
    image_size = 256
    cropped_size = 224
    num_class = 1000
    dataset_root = '/path/to/your/dataset'
    dataloader_workers = 8
    shuffle = True
    train_batch = 128
    test_batch = 64
    should_download = True

    # train parameters
    num_epoch = 500
    base_lr = 0.0001
    lr_decay = 0.98
    momentum = 0.9
    dropout_prob = 0.2
    weight_decay = 0.0005
    should_save = True
    should_test = True
    test_every = 1

    # model restore parameters
    resume_from = None  # None for train from scratch
    pre_trained_from = None # None for train from scratch
    save_every = 100

    def __init__(self):
        # create training dir
        self.summary_dir, self.ckpt_dir = create_train_dir(self)

# redefine cifar parameters
class CIFAR10_params(Params):
    dataset = 'cifar10'
    s = [1, 1, 1, 2, 2, 1, 2, 1]
    down_sample_rate = 8
    image_size = 32
    cropped_size = 32
    num_class = 10

class CIFAR100_params(CIFAR10_params):
    dataset = 'cifar100'
    num_class = 100

if __name__ == '__main__':
    c10 = CIFAR10_params()
    c100 = CIFAR100_params()
    print(c100.num_class)
