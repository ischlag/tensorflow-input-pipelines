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
OUTDIR="../../data/imagenet"
echo "Saving imagenet data to $OUTDIR"
mkdir -p "${OUTDIR}"
wget -nc "http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_bbox_train_v2.tar.gz" -O "${OUTDIR}/bounding_boxes/annotations.tar.gz"

# extract bounding box annotations
tar xzf "${OUTDIR}/bounding_boxes/annotations.tar.gz" -C "${OUTDIR}/bounding_boxes"

# download validation data
wget -nd -c "http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_img_val.tar" -O "${OUTDIR}/ILSVRC2012_img_val.tar"

# extract validation data
mkdir -p "${OUTDIR}/validation/"
tar xf "${OUTDIR}/ILSVRC2012_img_val.tar" -C "${OUTDIR}/validation/"

# download train data
wget -nd -c "http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_img_train.tar" -O "${OUTDIR}/ILSVRC2012_img_train.tar"

# extract individual train tar files
SYNSET_FILE="${OUTDIR}/imagenet_lsvrc_2015_synsets.txt"
wget -N "https://raw.githubusercontent.com/tensorflow/models/master/inception/inception/data/imagenet_lsvrc_2015_synsets.txt" -O "${SYNSET_FILE}"
while read SYNSET; do
  echo "Processing: ${SYNSET}"
  mkdir -p "${OUTDIR}/train/${SYNSET}"
  rm -rf "${OUTDIR}/train/${SYNSET}/*"

  tar xf "${OUTDIR}/ILSVRC2012_img_train.tar" "${SYNSET}.tar"
  tar xf "${SYNSET}.tar" -C "${OUTDIR}/train/${SYNSET}/"
  rm -f "${SYNSET}.tar"

  echo "Finished processing: ${SYNSET}"
done < "${SYNSET_FILE}"

# put validation data into directories just as the training data
wget -N "https://raw.githubusercontent.com/tensorflow/models/master/inception/inception/data/imagenet_2012_validation_synset_labels.txt" -O "${OUTDIR}/imagenet_2012_validation_synset_labels.txt"
python preprocess_imagenet_validation_data.py "${OUTDIR}/validation/" "${OUTDIR}/imagenet_2012_validation_synset_labels.txt"

# extract bounding box infor into an xml file
python process_bounding_boxes.py "${OUTDIR}/bounding_boxes/" "${OUTDIR}/imagenet_lsvrc_2015_synsets.txt"
mv imagenet_2012_bounding_boxes.csv "${OUTDIR}/imagenet_2012_bounding_boxes.csv"
# download the metadata text file
wget -N "https://raw.githubusercontent.com/tensorflow/models/master/inception/inception/data/imagenet_metadata.txt" -O "${OUTDIR}/imagenet_metadata.txt"

