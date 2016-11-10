import scipy.io
import numpy as np
from utils import download

########################################################################

# Directory where you want to download and save the data-set.
# Set this before you start calling any of the functions below.
data_path = "data/stanford_cars/"

# URL for the data-set on the internet.
data_url = "http://imagenet.stanford.edu/internal/car196/car_ims.tgz"
mat_url = "http://imagenet.stanford.edu/internal/car196/cars_annos.mat"

label_file = "cars_annos.mat"
num_classes = 196

def download_data():
  download.maybe_download_and_extract(url=data_url, download_dir=data_path)
  download.maybe_download(url=mat_url, download_dir=data_path)


def load_class_names():
  label_data = scipy.io.loadmat(data_path + label_file)
  return np.array([q[0] for q in label_data["class_names"][0]])


def load_training_data():
  label_data = scipy.io.loadmat(data_path + label_file)
  set = 1
  data = [(data_path + q[0][0], int(q[1][0][0]), int(q[2][0][0]), int(q[3][0][0]), int(q[4][0][0]), int(q[5][0][0])-1)
                for q in label_data["annotations"][0] if q[6][0][0]==set]

  img_path, bbox_x1, bbox_y1, bbox_x2, bbox_y2, cls = zip(*data)
  img_path, bbox_x1, bbox_y1, bbox_x2, bbox_y2, cls = np.array(img_path), np.array(bbox_x1), np.array(bbox_y1), np.array(bbox_x2), np.array(bbox_y2), np.array(cls)

  return img_path, bbox_x1, bbox_y1, bbox_x2, bbox_y2, cls, download.one_hot_encoded(class_numbers=cls, num_classes=num_classes)


def load_test_data():
  label_data = scipy.io.loadmat(data_path + label_file)
  set = 0
  data = [(data_path + q[0][0], int(q[1][0][0]), int(q[2][0][0]), int(q[3][0][0]), int(q[4][0][0]), int(q[5][0][0])-1)
                for q in label_data["annotations"][0] if q[6][0][0]==set]

  img_path, bbox_x1, bbox_y1, bbox_x2, bbox_y2, cls = zip(*data)
  img_path, bbox_x1, bbox_y1, bbox_x2, bbox_y2, cls = np.array(img_path), np.array(bbox_x1), np.array(bbox_y1), np.array(bbox_x2), np.array(bbox_y2), np.array(cls)

  return img_path, bbox_x1, bbox_y1, bbox_x2, bbox_y2, cls, download.one_hot_encoded(class_numbers=cls, num_classes=num_classes)






