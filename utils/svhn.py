###############################################################################
# Author:       Imanol Schlag (more info on ischlag.github.io)
# Description:  Functions to download and load SVHN into memory.
# Date:         10.11.2016
#
#

import sys
import os
from six.moves.urllib.request import urlretrieve

from utils.download import one_hot_encoded
import scipy.io

data_path = "data/SVHN/"

data_url = 'http://ufldl.stanford.edu/housenumbers/'
train_data = 'train_32x32.mat'
test_data = 'test_32x32.mat'
extra_data = 'extra_32x32.mat'

num_classes = 10

last_percent_reported = None

def __download_progress_hook(count, blockSize, totalSize):
  """A hook to report the progress of a download. This is mostly intended for users with
  slow internet connections. Reports every 1% change in download progress.
  """

  global last_percent_reported
  percent = int(count * blockSize * 100 / totalSize)

  if last_percent_reported != percent:
    if percent % 5 == 0:
      sys.stdout.write("%s%%" % percent)
      sys.stdout.flush()
    else:
      sys.stdout.write(".")
      sys.stdout.flush()

    last_percent_reported = percent


def __maybe_download(filename, force=False):
  """Download a file if not present, and make sure it's the right size."""

  if force or not os.path.exists(data_path + filename):
    if force or not os.path.exists(data_path):
      os.makedirs(data_path)

    print('Attempting to download:', filename)
    filename, _ = urlretrieve(data_url + filename, data_path + filename, reporthook=__download_progress_hook)
    print('\nDownload Complete!')
  else:
    print('skipping ', filename)
  return filename


def download_data():
  """Download the SVHN data if it doesn't exist yet."""

  __maybe_download(train_data)
  __maybe_download(test_data)
  __maybe_download(extra_data)

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

  return images, cls, one_hot_encoded(class_numbers=cls, num_classes=num_classes)

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

  return images, cls, one_hot_encoded(class_numbers=cls, num_classes=num_classes)


def load_extra_data():
  extra_data = scipy.io.loadmat(data_path + 'extra_32x32.mat', variable_names='X').get('X')
  extra_labels = scipy.io.loadmat(data_path + 'extra_32x32.mat', variable_names='y').get('y')

  images = extra_data.transpose((3,0,1,2)) / 255.0
  cls = extra_labels[:, 0]
  cls[cls == 10] = 0

  return images, cls, one_hot_encoded(class_numbers=cls, num_classes=num_classes)


