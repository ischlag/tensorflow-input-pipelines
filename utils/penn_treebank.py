###############################################################################
# Author:       Imanol Schlag (more info on ischlag.github.io)
# Description:  downloads and loads the penn treebank dataset into memory.
# Date:         11.2016
#
#

import os
import tensorflow as tf
import collections

from utils import download

########################################################################

# Directory where you want to download and save the data-set.
# Set this before you start calling any of the functions below.
data_path = "data/penn_treebank/"

# URL for the data-set on the internet.
data_url = "http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz"

train_data_file = data_path + "ptb.char.train.txt"
test_data_file = data_path + "ptb.char.test.txt"
valid_data_file = data_path + "ptb.char.valid.txt"

########################################################################
# some useful functions from TensorFlow

def _read_words(filename):
  with tf.gfile.GFile(filename, "r") as f:
    return f.read().decode("utf-8").replace("\n", "<eos>").split()

def _build_vocab(filename):
  data = _read_words(filename)

  counter = collections.Counter(data)
  count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

  words, _ = list(zip(*count_pairs))
  word_to_id = dict(zip(words, range(len(words))))

  return word_to_id

def _file_to_word_ids(filename, word_to_id):
  data = _read_words(filename)
  return [word_to_id[word] for word in data if word in word_to_id]

########################################################################

def download_data():
  """Download the penn treebank data if it doesn't exist yet."""
  if download.maybe_download_and_extract(url=data_url, download_dir=data_path):
    os.system("mv " + data_path + "/simple-examples/data/* " + data_path)
    os.system("rm -r " + data_path + "/simple-examples")

def load_training_data():
  word_to_id = _build_vocab(train_data_file)
  train_data = _file_to_word_ids(train_data_file, word_to_id)
  vocabulary = len(word_to_id)
  return train_data, vocabulary

def load_test_data():
  word_to_id = _build_vocab(train_data_file)
  train_data = _file_to_word_ids(test_data_file, word_to_id)
  vocabulary = len(word_to_id)
  return train_data, vocabulary

def load_validation_data():
  word_to_id = _build_vocab(train_data_file)
  train_data = _file_to_word_ids(valid_data_file, word_to_id)
  vocabulary = len(word_to_id)
  return train_data, vocabulary

