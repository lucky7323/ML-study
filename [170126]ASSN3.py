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
batch_size = 500
epochs = 15
learning_rate = 0.001
drop_rate = 0.3
hidden_size = 256

# falttening x & convert one-hot vector y
x = train_images.reshape([60000, 28*28])
y_ = np.zeros((60000,10))
y_[np.arange(60000), train_labels] = 1

# weights and bias
W1 = np.random.rand(784,hidden_size)
W2 = np.random.rand(hidden_size,hidden_size)
W3 = np.random.rand(hidden_size,10)

B1 = np.random.rand(hidden_size)
B2 = np.random.rand(hidden_size)
B3 = np.random.rand(10)

# plotting variable
arr = np.zeros(epochs)

# batch function
def batch(x, y, n):
    ran = n * (np.random.random_integers(1,(60000/n)) - 1)
    return x[ran + np.arange(n)], y[ran + np.arange(n)]

# logsoftmax function
def lsoftmax(x):
    z = x-x.max(1,keepdims=True)
    return (z - np.log(np.sum(np.exp(z),axis=1,keepdims=True)))

# cross-entropy function
def cross_entropy(batch_xs, batch_ys, W1, W2, W3, B1, B2, B3):
    H1 = np.maximum(0, np.matmul(batch_xs, W1) + B1)
    D1 = np.random.rand(*H1.shape) > drop_rate
    H1 *= D1
    H2 = np.maximum(0, np.matmul(H1, W2) + B2)
    D2 = np.random.rand(*H2.shape) > drop_rate
    H2 *= D2
    y = np.matmul(H2, W3) + B3
    return np.mean(-np.sum(batch_ys * lsoftmax(y), axis=1))

# gradient function
grad_fun_W1 = grad(cross_entropy, argnum=2)
grad_fun_W2 = grad(cross_entropy, argnum=3)
grad_fun_W3 = grad(cross_entropy, argnum=4)
grad_fun_B1 = grad(cross_entropy, argnum=5)
grad_fun_B2 = grad(cross_entropy, argnum=6)
grad_fun_B3 = grad(cross_entropy, argnum=7)

# training
for i in range(epochs):
    batch_xs, batch_ys = batch(x, y_, batch_size)
    W1 -= grad_fun_W1(batch_xs, batch_ys, W1, W2, W3, B1, B2, B3) * learning_rate
    W2 -= grad_fun_W2(batch_xs, batch_ys, W1, W2, W3, B1, B2, B3) * learning_rate
    W3 -= grad_fun_W3(batch_xs, batch_ys, W1, W2, W3, B1, B2, B3) * learning_rate
    B1 -= grad_fun_B1(batch_xs, batch_ys, W1, W2, W3, B1, B2, B3) * learning_rate
    B2 -= grad_fun_B2(batch_xs, batch_ys, W1, W2, W3, B1, B2, B3) * learning_rate
    B3 -= grad_fun_B3(batch_xs, batch_ys, W1, W2, W3, B1, B2, B3) * learning_rate
    arr[i] = cross_entropy(batch_xs, batch_ys, W1,W2,W3,B1,B2,B3)

# plotting loss function graph
plt.plot(arr, 'r--')
plt.show()

# evaluating
x_test = test_images.reshape([10000, 28*28])
y_answer = np.zeros((10000,10))
y_answer[np.arange(10000), test_labels] = 1

H1_ = np.maximum(0, np.matmul(x_test, W1) + B1)
H2_ = np.maximum(0, np.matmul(H1_, W2) + B2)
y_predict = np.matmul(H2_, W3) + B3

accuracy = np.mean(np.equal(np.argmax(y_predict,1), np.argmax(y_answer,1))) * 100
print("Accuracy : %0.2f%%" % accuracy)


