import os
import gzip
import struct
import array
import autograd.numpy as np
import matplotlib.pyplot as plt
from autograd import grad
from urllib.request import urlretrieve

# mnist download
def download(url, filename):
    if not os.path.exists('data'):
        os.makedirs('data')
    out_file = os.path.join('data', filename)
    if not os.path.isfile(out_file):
        urlretrieve(url, out_file)

base_url = 'http://yann.lecun.com/exdb/mnist/'

def parse_labels(filename):
    with gzip.open(filename, 'rb') as fh:
        magic, num_data = struct.unpack(">II", fh.read(8))
        return np.array(array.array("B", fh.read()), dtype=np.uint8)

def parse_images(filename):
    with gzip.open(filename, 'rb') as fh:
        magic, num_data, rows, cols = struct.unpack(">IIII", fh.read(16))
        return np.array(array.array("B", fh.read()), dtype=np.uint8).reshape(num_data, rows, cols)

for filename in ['train-images-idx3-ubyte.gz',
                 'train-labels-idx1-ubyte.gz',
                 't10k-images-idx3-ubyte.gz',
                 't10k-labels-idx1-ubyte.gz']:
    download(base_url + filename, filename)

train_images = parse_images('data/train-images-idx3-ubyte.gz')
train_labels = parse_labels('data/train-labels-idx1-ubyte.gz')
test_images  = parse_images('data/t10k-images-idx3-ubyte.gz')
test_labels  = parse_labels('data/t10k-labels-idx1-ubyte.gz')

# parameters
learning_rate = 0.5
batch_size = 100
epoch = 1000

# data flattening & convert to one-hot vector
x = train_images.reshape([60000, 28*28])
y_ = np.zeros((60000,10))
y_[np.arange(60000), train_labels] = 1

# logsoftmax function (prevent over/under flow)
def lsoftmax(x):
    z = x-x.max(1,keepdims=True)
    return (z - np.log(np.sum(np.exp(z),axis=1,keepdims=True)))

# cross-entropy
def cross_entropy(W, b):            
    ran = batch_size * (np.random.random_integers(1,60000/batch_size) - 1)
    x_sample = x[ran + np.arange(batch_size)]
    y_sample = y_[ran + np.arange(batch_size)]
    x_sample = x_sample / 255.0
    y = np.matmul(x_sample,W) + b
    return np.mean(-np.sum(y_sample * lsoftmax(y), axis=1))

# gradient function
grad_fun_W = grad(cross_entropy, argnum=0)
grad_fun_b = grad(cross_entropy, argnum=1)

# weight and bias initialization
W = np.zeros((784,10))
b = np.zeros(10)

# plotting variable
arr = np.zeros(epoch) 

# training
for i in range(epoch):
    W -= grad_fun_W(W,b) * learning_rate
    b -= grad_fun_b(W,b) * learning_rate
    arr[i] = cross_entropy(W,b)
    if (i+1)%100 == 0:
        # evaluating
        x_test = test_images.reshape([10000, 28*28])
        x_test = x_test / 255.0
        y_answer = np.zeros((10000,10))
        y_answer[np.arange(10000), test_labels] = 1
        y_predict = np.matmul(x_test,W) + b
        accuracy = np.mean(np.equal(np.argmax(y_predict,1), np.argmax(y_answer,1))) * 100
        print('{} epoch: {:.2f}, cost: {:.2f}'.format(i+1, accuracy, arr[i]))
        if (i+1)==epoch:
            print('Accuracy : {:.2f}'.format(accuracy))

# plotting cost function value
plt.plot(arr, 'r--')
plt.show()

