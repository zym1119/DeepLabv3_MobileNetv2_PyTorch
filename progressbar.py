import time


class bar(object):
    def __init__(self):
        self.start_time = None
        self.iter_per_sec = 0
        self.time = None

    def click(self, current_idx, max_idx, total_length=40):
        """
        Each click is a draw procedure of progressbar
        :param current_idx: range from 0 to max_idx-1
        :param max_idx: maximum iteration
        :param total_length: length of progressbar
        """
        if self.start_time is None:
            self.start_time = time.time()
        else:
            self.time = time.time()-self.start_time
            self.iter_per_sec = 1/self.time
            perc = current_idx * total_length // max_idx
            # print progress bar
            print('\r|'+'='*perc+'>'+' '*(total_length-1-perc)+'| %d/%d (%.2f iter/s)' % (current_idx+1,
                                                                                          max_idx,
                                                                                          self.iter_per_sec), end='')
            self.start_time = time.time()

    def close(self):
        self.__init__()
        print('')

if __name__ == '__main__':
    pb = bar()
    for i in range(10):
        pb.click(i, 10)
        time.sleep(0.5)
        print(pb.time)
    pb.close()
