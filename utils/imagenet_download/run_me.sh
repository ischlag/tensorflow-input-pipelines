# Download, extract, and preprocess the imagenet data using TensorFlows imagenet scripts
# The original script didn't work for me because of different reasons.
# Imanol Schlag, 11.2016
# original: https://github.com/tensorflow/models/blob/master/inception/inception/data/download_imagenet.sh
#
# Size Info:
# ILSVRC2012_img_train.tar is about 147.9 GB
# ILSVRC2012_img_val.tar is about 6.7 GB
# bounding_boxes/ is about 324.5 MB
#
# Usage:
# Copy this shell script and all python scripts in the same folder into your imagenet data folder.
# Run this shell script.

# download bounding boxes
wget "http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_bbox_train_v2.tar.gz" -O "./bounding_boxes/annotations.tar.gz"

# extract bounding box annotations
tar xzf "./bounding_boxes/annotations.tar.gz" -C "./bounding_boxes"

# download validation data
wget -nd -c "http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_img_val.tar"

# extract validation data
tar xf "ILSVRC2012_img_val.tar" -C "./validation/"

# download train data
wget -nd -c "http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_img_train.tar"

# extract individual train tar files
wget -nc "https://github.com/tensorflow/models/blob/master/inception/inception/data/imagenet_lsvrc_2015_synsets.txt"
SYNSET_FILE="imagenet_lsvrc_2015_synsets.txt"
while read SYNSET; do
  echo "Processing: ${SYNSET}"
  mkdir -p "./train/${SYNSET}"
  rm -rf "./train/${SYNSET}/*"

  tar xf "ILSVRC2012_img_train.tar" "${SYNSET}.tar"
  tar xf "${SYNSET}.tar" -C "./train/${SYNSET}/"
  rm -f "${SYNSET}.tar"

  echo "Finished processing: ${SYNSET}"
done < "${SYNSET_FILE}"

# put validation data into directories just as the training data
wget -nc "https://github.com/tensorflow/models/blob/master/inception/inception/data/imagenet_2012_validation_synset_labels.txt"
python preprocess_imagenet_validation_data.py "./validation/" "imagenet_2012_validation_synset_labels.txt"

# extract bounding box infor into an xml file
python process_bounding_boxes.py "./bounding_boxes/" "imagenet_lsvrc_2015_synsets.txt"

# download the metadata text file
wget -nc "https://github.com/tensorflow/models/blob/master/inception/inception/data/imagenet_metadata.txt"

