
import os

from keras.layers import *
from keras.losses import *
from keras.optimizers import *
from keras.models import Model

from tqdm import tqdm

from utils import MovingAvg


class ModelBase():

    def __init__(self, name, loader):
        self._name = name
        self._model = None
        self._loader = loader

        self._epoch = 2
        self._metrics = []


    def create(self):
        # Override this
        pass


    def compile(self):
        # Override this
        pass


    # Load model for certain epoch
    def load(self, epoch_num, data_dir = 'weights'):
        if not os.path.exists(data_dir):
            os.mkdir(data_dir)

        fname = os.path.join(data_dir, '%s_%d.hdf5' % (self._name, epoch_num))

        if (os.path.exists(fname)):
            self._model.load_weights(fname)
            self._epoch = epoch_num
        else:
            print('Model file', fname, 'does not exist')

        self._epoch = epoch_num


    def train(self, num_epochs, mb_size, test_freq, save_dir = 'weights'):
        start = self._epoch

        # Calculate minibatches per epoch
        mb_per_epoch = self._loader.get_epoch_size() // mb_size

        for epoch in range(start, num_epochs + 1):
            self._epoch = epoch
            print('Epoch', 1)

            # Average values
            avgs = [MovingAvg(100)] * len(self._metrics)
            total_avgs = [0.0] * len(self._metrics)

            progress = tqdm(range(mb_per_epoch))

            for mb in progress:
                # Train
                metrics = self._train_mb(mb_size)

                # Update metrics
                prog_label = ''
                for i in len(metrics):
                    avgs[i].add(metrics[i])
                    total_avgs[i] += metrics[i]

                    prog_label += '%s: %.3f | ' % (self._metrics[i], avgs[i].value())

                progress.set_description(prog_label)

            # Save weights
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            fname = os.path.join(save_dir, '%s_%d.hdf5' % (self._name, epoch))
            self._model.save_weights(fname)


    # Given the minibatch size, do one iteration of training, and return targeted metrics
    def _train_mb(self, mb_size):
        # Override this
        return []