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

### Installation and Running the Cifar-100 Example
```
schlag@box:~/MyStuff/input_pipelines$ mkvirtualenv $(pwd | awk '{print $1"/env"}')
Using base prefix '/usr'
New python executable in /home/schlag/MyStuff/input_pipelines/env/bin/python3
Also creating executable in /home/schlag/MyStuff/input_pipelines/env/bin/python
Installing setuptools, pip, wheel...done.
schlag@box:~/MyStuff/input_pipelines$ source env/bin/activate
(env) schlag@box:~/MyStuff/input_pipelines$ pip3 install -r pip3_requirements.txt 
Collecting numpy==1.11.2 (from -r pip3_requirements.txt (line 1))
  Using cached numpy-1.11.2-cp35-cp35m-manylinux1_x86_64.whl
Collecting pickleshare==0.7.4 (from -r pip3_requirements.txt (line 2))
  Using cached pickleshare-0.7.4-py2.py3-none-any.whl
Collecting protobuf==3.0.0 (from -r pip3_requirements.txt (line 3))
  Using cached protobuf-3.0.0-py2.py3-none-any.whl
Collecting scipy==0.18.1 (from -r pip3_requirements.txt (line 4))
  Using cached scipy-0.18.1-cp35-cp35m-manylinux1_x86_64.whl
Collecting six==1.10.0 (from -r pip3_requirements.txt (line 5))
  Using cached six-1.10.0-py2.py3-none-any.whl
Requirement already satisfied: setuptools in ./env/lib/python3.5/site-packages (from protobuf==3.0.0->-r pip3_requirements.txt (line 3))
Installing collected packages: numpy, pickleshare, six, protobuf, scipy
Successfully installed numpy-1.11.2 pickleshare-0.7.4 protobuf-3.0.0 scipy-0.18.1 six-1.10.0
(env) schlag@box:~/MyStuff/input_pipelines$ pip3 install ../tf-builds/tensorflow-0.11.0rc2-cp35-cp35m-linux_x86_64.whl 
Processing /home/schlag/MyStuff/tf-builds/tensorflow-0.11.0rc2-cp35-cp35m-linux_x86_64.whl
Requirement already satisfied: wheel>=0.26 in ./env/lib/python3.5/site-packages (from tensorflow==0.11.0rc2)
Requirement already satisfied: six>=1.10.0 in ./env/lib/python3.5/site-packages (from tensorflow==0.11.0rc2)
Collecting protobuf==3.1.0 (from tensorflow==0.11.0rc2)
  Using cached protobuf-3.1.0-py2.py3-none-any.whl
Requirement already satisfied: numpy>=1.11.0 in ./env/lib/python3.5/site-packages (from tensorflow==0.11.0rc2)
Requirement already satisfied: setuptools in ./env/lib/python3.5/site-packages (from protobuf==3.1.0->tensorflow==0.11.0rc2)
Installing collected packages: protobuf, tensorflow
  Found existing installation: protobuf 3.0.0
    Uninstalling protobuf-3.0.0:
      Successfully uninstalled protobuf-3.0.0
Successfully installed protobuf-3.1.0 tensorflow-0.11.0rc2
(env) schlag@box:~/MyStuff/input_pipelines$ python cifar-100_example.py 
I tensorflow/stream_executor/dso_loader.cc:128] successfully opened CUDA library libcublas.so.8.0.27 locally
I tensorflow/stream_executor/dso_loader.cc:128] successfully opened CUDA library libcudnn.so.5.1.5 locally
I tensorflow/stream_executor/dso_loader.cc:128] successfully opened CUDA library libcufft.so.8.0.27 locally
I tensorflow/stream_executor/dso_loader.cc:128] successfully opened CUDA library libcuda.so.1 locally
I tensorflow/stream_executor/dso_loader.cc:128] successfully opened CUDA library libcurand.so.8.0.27 locally
I tensorflow/core/common_runtime/gpu/gpu_device.cc:885] Found device 0 with properties: 
name: GeForce GTX 1080
major: 6 minor: 1 memoryClockRate (GHz) 1.7335
pciBusID 0000:05:00.0
Total memory: 7.92GiB
Free memory: 6.63GiB
I tensorflow/core/common_runtime/gpu/gpu_device.cc:906] DMA: 0 
I tensorflow/core/common_runtime/gpu/gpu_device.cc:916] 0:   Y 
I tensorflow/core/common_runtime/gpu/gpu_device.cc:975] Creating TensorFlow device (/gpu:0) -> (device: 0, name: GeForce GTX 1080, pci bus id: 0000:05:00.0)
Loading CIFAR-100 data
- Download progress: 100.0%
Download finished. Extracting files.
Extracting finished. Cleaning up.
Done.
Loading data: data/CIFAR-100/cifar-100-python/train
batch  0
(256, 32, 32, 3)
(256, 100)
batch  1
(256, 32, 32, 3)
(256, 100)
batch  2
(256, 32, 32, 3)
(256, 100)
batch  3
(256, 32, 32, 3)
(256, 100)
batch  4
(256, 32, 32, 3)
(256, 100)
batch  5
(256, 32, 32, 3)
(256, 100)
batch  6
(256, 32, 32, 3)
(256, 100)
batch  7
(256, 32, 32, 3)
(256, 100)
batch  8
(256, 32, 32, 3)
(256, 100)
batch  9
(256, 32, 32, 3)
(256, 100)
done!

```


### Download the Imagenet Data
You need to use the supplied shell script in order to download the imagenet data. This can take a long time. The train archive is almost 150GB in size.

```
(env) schlag@box:~/MyStuff/input_pipelines$ cd utils/imagenet_download/
(env) schlag@box:~/MyStuff/input_pipelines/utils/imagenet_download$ sh run_me.sh
** snip (this will take a while)  **
(env) schlag@box:~/MyStuff/input_pipelines/utils/imagenet_download$ cd ../../
(env) schlag@box:~/MyStuff/input_pipelines$ python imagenet_example.py
I tensorflow/stream_executor/dso_loader.cc:128] successfully opened CUDA library libcublas.so.8.0.27 locally
I tensorflow/stream_executor/dso_loader.cc:128] successfully opened CUDA library libcudnn.so.5.1.5 locally
I tensorflow/stream_executor/dso_loader.cc:128] successfully opened CUDA library libcufft.so.8.0.27 locally
I tensorflow/stream_executor/dso_loader.cc:128] successfully opened CUDA library libcuda.so.1 locally
I tensorflow/stream_executor/dso_loader.cc:128] successfully opened CUDA library libcurand.so.8.0.27 locally
I tensorflow/core/common_runtime/gpu/gpu_device.cc:885] Found device 0 with properties: 
name: GeForce GTX 1080
major: 6 minor: 1 memoryClockRate (GHz) 1.7335
pciBusID 0000:05:00.0
Total memory: 7.92GiB
Free memory: 6.61GiB
I tensorflow/core/common_runtime/gpu/gpu_device.cc:906] DMA: 0 
I tensorflow/core/common_runtime/gpu/gpu_device.cc:916] 0:   Y 
I tensorflow/core/common_runtime/gpu/gpu_device.cc:975] Creating TensorFlow device (/gpu:0) -> (device: 0, name: GeForce GTX 1080, pci bus id: 0000:05:00.0)
Successfully read 615299 bounding boxes across 544546 images.
Determining list of input files and labels from data/imagenet/validation/.
Finished finding files in 100 of 1000 classes.
Finished finding files in 200 of 1000 classes.
Finished finding files in 300 of 1000 classes.
Finished finding files in 400 of 1000 classes.
Finished finding files in 500 of 1000 classes.
Finished finding files in 600 of 1000 classes.
Finished finding files in 700 of 1000 classes.
Finished finding files in 800 of 1000 classes.
Finished finding files in 900 of 1000 classes.
Finished finding files in 1000 of 1000 classes.
Found 50000 JPEG files across 1000 labels inside data/imagenet/validation/.
Determining list of input files and labels from data/imagenet/train/.
Finished finding files in 100 of 1000 classes.
Finished finding files in 200 of 1000 classes.
Finished finding files in 300 of 1000 classes.
Finished finding files in 400 of 1000 classes.
Finished finding files in 500 of 1000 classes.
Finished finding files in 600 of 1000 classes.
Finished finding files in 700 of 1000 classes.
Finished finding files in 800 of 1000 classes.
Finished finding files in 900 of 1000 classes.
Finished finding files in 1000 of 1000 classes.
Found 1281167 JPEG files across 1000 labels inside data/imagenet/train/.
Loading imagenet data
Train directory seems to exist
Validation directory seems to exist
batch  0
(64, 299, 299, 3)
(64, 1000)
batch  1
(64, 299, 299, 3)
(64, 1000)
batch  2
(64, 299, 299, 3)
(64, 1000)
batch  3
(64, 299, 299, 3)
(64, 1000)
batch  4
(64, 299, 299, 3)
(64, 1000)
batch  5
(64, 299, 299, 3)
(64, 1000)
batch  6
(64, 299, 299, 3)
(64, 1000)
batch  7
(64, 299, 299, 3)
(64, 1000)
batch  8
(64, 299, 299, 3)
(64, 1000)
batch  9
(64, 299, 299, 3)
(64, 1000)
done!

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

