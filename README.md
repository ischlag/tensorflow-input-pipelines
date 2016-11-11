# TensorFlow Input Pipelines

Use these TensorFlow(v0.11) pipelines to automatically download and easily fetch batches of data and labels from some of the most used datasets in Deep Learning. The implementations are threaded, efficient, can be randomized and also include large datasets such as imagenet. 

### Supported Datasets
- MNIST
- CIFAR-10
- CIFAR-100
- SVHN
- Stanford Cars 196
- Imagenet (no automatic data download, but a shell script is provided in utils/imagenet_download/)
- Penn Treebank

(more datasets will be added soon ...)

### Example
```python
import tensorflow as tf
sess = tf.Session()

with tf.device('/cpu:0'):
  from datasets.svhn import svhn_data
  d = svhn_data(batch_size=256, sess=sess)
  image_batch_tensor, target_batch_tensor = d.build_train_data_tensor()

for i in range(5):
  print("batch ", i)
  image_batch, target_batch = sess.run([image_batch_tensor, target_batch_tensor])
  # logits = model(image_batch, target_batch)
  # ...
  print(image_batch.shape)
  print(target_batch.shape)
  
d.close()
sess.close()
```

### Installation
```
mkvirtualenv env
git clone https://github.com/ischlag/tensorflow-input-pipelines.git
cd tensorflow-input-pipelines
(env) pip3 install -r pip3_requirements.txt
# install TensorFlow yourself ...
python example_train.py
```

### Train Script Template
A CNN training script template is provided with the following features:
- easy switchin of datasets
- separate training and testing streams
- continous console log 
- test-set evaluation after every epoch
- automatically saves the best performing model parameters
- automatically decreases the learning rate after if there is no improvement in accuracy
- evaluate top 1 and top n accuracies
- easy parameter loading from a previous save point to continue training
- prints a confusion matrix in your console

