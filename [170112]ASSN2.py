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

# convert to one-hot vector
y_ = np.zeros((60000,10))
y_[np.arange(60000), train_labels] = 1

# logsoftmax function (prevent over/under flow)
def lsoftmax(x):
    z = x-x.max(1,keepdims=True)
    return (z - np.log(np.sum(np.exp(z),axis=1,keepdims=True)))

# falttening
x = train_images.reshape([60000, 28*28])

# cross-entropy (random)
def cross_entropy(W, b):            
    ran = 100 * (np.random.random_integers(1,600) - 1)
    x_sample = x[ran + np.arange(100)]
    y_sample = y_[ran + np.arange(100)]
    y = np.matmul(x_sample,W) + b
    return np.mean(-np.sum(y_sample * lsoftmax(y), axis=1))

# cross-entropy (sequence)
def cross_entropy_(W, b, n):
    x_sample = x[n*100 + np.arange(200)]
    y_sample = y_[n*100 + np.arange(200)]
    y = np.matmul(x_sample,W) + b
    return np.mean(-np.sum(y_sample * lsoftmax(y), axis=1))

grad_fun_W = grad(cross_entropy, argnum=0)
grad_fun_b = grad(cross_entropy, argnum=1)
grad_fun_W1 = grad(cross_entropy_, argnum=0)
grad_fun_b1 = grad(cross_entropy_, argnum=1)

W0 = np.zeros((784,10))
b0 = np.zeros(10)
W1 = np.zeros((784,10))
b1 = np.zeros(10)
W2 = np.zeros((784,10))
b2 = np.zeros(10)

arr1 = np.zeros(11) 
arr2 = np.zeros(11)

#training (random)
for i in range(1000):
    W1 -= grad_fun_W(W1,b0) * 0.5
    b1 -= grad_fun_b(W0,b1) * 0.5
    if i%100==0 or i==1000:
        arr1[int(i/100)] = cross_entropy(W1,b1)
plt.plot(arr1, 'r--')
plt.show()

#training (sequence)
for i in range(1000):
    W2 -= grad_fun_W1(W2,b0,i%300) * 0.5
    b2 -= grad_fun_b1(W0,b2,i%300) * 0.5
    if i%100==0 or i==1000:
        arr2[int(i/100)] = cross_entropy(W2,b2)
plt.plot(arr2, 'r--')
plt.show()

#evaluating
x_test = test_images.reshape([10000, 28*28])
y_answer = np.zeros((10000,10))
y_answer[np.arange(10000), test_labels] = 1
y_predict1 = np.matmul(x_test,W1) + b1
y_predict2 = np.matmul(x_test,W2) + b2
accuracy1 = np.mean(np.equal(np.argmax(y_predict1,1), np.argmax(y_answer,1))) * 100
accuracy2 = np.mean(np.equal(np.argmax(y_predict2,1), np.argmax(y_answer,1))) * 100

print("Accuracy (random) : %0.2f%%" % accuracy1)
print("Accuracy (order) : %0.2f%%" % accuracy2)

