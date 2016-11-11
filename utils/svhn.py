###############################################################################
# Author:       Imanol Schlag (more info on ischlag.github.io)
# Description:  Functions to download and load SVHN into memory.
# Date:         10.11.2016
#
#

from utils import download
import scipy.io

data_path = "data/SVHN/"

data_url = 'http://ufldl.stanford.edu/housenumbers/'
train_data = 'train_32x32.mat'
test_data = 'test_32x32.mat'
extra_data = 'extra_32x32.mat'

num_classes = 10

def download_data():
  """Download the SVHN data if it doesn't exist yet."""

  download.maybe_download(url=data_url + train_data, download_dir=data_path)
  download.maybe_download(url=data_url + test_data, download_dir=data_path)
  download.maybe_download(url=data_url + extra_data, download_dir=data_path)

def load_training_data():
  """
  Load all the training-data for the SVHN data-set.
  Returns the images, class-numbers and one-hot encoded class-labels.
  """

  train_data = scipy.io.loadmat(data_path + 'train_32x32.mat', variable_names='X').get('X')
  train_labels = scipy.io.loadmat(data_path + 'train_32x32.mat', variable_names='y').get('y')

  images = train_data.transpose((3,0,1,2)) / 255.0
  cls = train_labels[:, 0]
  cls[cls == 10] = 0

  return images, cls, download.one_hot_encoded(class_numbers=cls, num_classes=num_classes)

def load_test_data():
  """
  Load all the test-data for the SVHN data-set.
  Returns the images, class-numbers and one-hot encoded class-labels.
  """

  test_data = scipy.io.loadmat(data_path + 'test_32x32.mat', variable_names='X').get('X')
  test_labels = scipy.io.loadmat(data_path + 'test_32x32.mat', variable_names='y').get('y')

  images = test_data.transpose((3, 0, 1, 2)) / 255.0
  cls = test_labels[:, 0]
  cls[cls == 10] = 0

  return images, cls, download.one_hot_encoded(class_numbers=cls, num_classes=num_classes)


def load_extra_data():
  extra_data = scipy.io.loadmat(data_path + 'extra_32x32.mat', variable_names='X').get('X')
  extra_labels = scipy.io.loadmat(data_path + 'extra_32x32.mat', variable_names='y').get('y')

  images = extra_data.transpose((3,0,1,2)) / 255.0
  cls = extra_labels[:, 0]
  cls[cls == 10] = 0

  return images, cls, download.one_hot_encoded(class_numbers=cls, num_classes=num_classes)


