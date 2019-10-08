
import os
from glob import glob
import random

import numpy as np

class LoaderBase():

    def __init__(self, data_dir, data_ext, dtype = np.float32):
        self._data_dir = data_dir
        self._data_ext = data_ext
        # Create data directory if needed
        if not os.path.exists(data_dir):
            os.mkdir(data_dir)

        # This is the data type used to store the loaded data
        self._dtype = dtype
        # Set this to override epoch size
        self._epoch_size_override = -1
        
        self._x_train = []
        self._y_train = []


    def load(self, num_to_load = -1):
        # Get all data files in directory
        fnames = glob(os.path.join(self._data_dir, '*.%s' % self._data_ext))
        # If custom number to load, choose random sample
        if num_to_load >= 0:
            fnames = random.sample(fnames, num_to_load)

        for fname in fnames:
            self._load_file(fname)

        self._x_train = np.array(self._x_train, dtype = self._dtype)
        self._y_train = np.array(self._y_train, dtype = self._dtype)


    # Given a file name, load data into x and y
    def _load_file(self, fname):
        # Override this loading function
        pass


    def get_batch(self, mb_size):
        idx = np.random.randint(self.get_epoch_size(), size = mb_size)
        return self._format_batch(idx)


    def get_epoch_size(self):
        if self._epoch_size_override >= 0:
            return self._epoch_size_override
        else:
            return len(self._x_train)


    # Given a set of indices, format features and labels for this batch
    def _format_batch(self, idx):
        # Override this
        pass