
import os
from glob import glob
import random

import numpy as np
from tqdm import tqdm

class LoaderBase():

	def __init__(self, data_dir, data_ext, dtype = np.float32):
		self._data_dir = data_dir
		self._data_ext = data_ext
		# Create data directory if needed
		if not os.path.exists(data_dir):
			os.mkdir(data_dir)

		# This is the data type used to store the loaded data
		self._dtype = dtype
		# Set this to override epoch size (To use smaller epoch size)
		self._epoch_size_override = -1
		
		self._x_train = []
		self._y_train = []
		self._x_test = []
		self._y_test = []


	def load(self, test_split, num_to_load = -1):
		# Get all data files in directory
		fnames = glob(os.path.join(self._data_dir, '*.%s' % self._data_ext))
		# If custom number to load, choose random sample
		if num_to_load >= 0:
			fnames = random.sample(fnames, num_to_load)

		print('Loading data...')
		for fname in tqdm(fnames):
			self._load_file(fname)

		# Return if nothing was loaded
		if len(self._x_train) == 0:
			return
			
		split_val = int(len(self._x_train) * test_split)
		self._x_train = np.array(self._x_train, dtype = self._dtype)
		self._y_train = np.array(self._y_train, dtype = self._dtype)
		# Create testing set
		self._x_test = self._x_train[-split_val:]

		# Only create labels testing set if labels have been provided
		if len(self._y_train) == len(self._x_train):
			self._y_test = self._y_train[-split_val:]

		# Update training sets
		self._x_train = self._x_train[:-split_val]
		self._y_train = self._y_train[:-split_val]


	# Given a file name, load data into x and y
	def _load_file(self, fname):
		# Override this loading function
		pass


	def get_training_batch(self, mb_size):
		idx = np.random.randint(self.get_epoch_size(), size = mb_size)
		return self._format_batch(idx, self._x_train, self._y_train)


	def get_testing_batch(self, mb_size):
		idx = np.random.randint(len(self._x_test), size = mb_size)
		return self._format_batch(idx, self._x_test, self._y_test)


	def get_epoch_size(self):
		if self._epoch_size_override >= 0:
			return self._epoch_size_override
		else:
			return len(self._x_train)


	# Given a set of indices, format and return data required for batch
	def _format_batch(self, idx, data_x, data_y):
		# Default implementation of format batch

		# Assumes that data_x and data_y have valid data
		features = data_x[idx].astype(np.float32)
		labels = data_y[idx].astype(np.float32)
		return (features, labels)