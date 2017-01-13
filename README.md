# ML-study
2016 winter ~ 2017 spring study with python

####[Assignment2 description]####
* The goal of this assignment is to make a simple linear classifer for MNIST, which contains 60,000 handwritten digit images with 10 classes.
* The most important thing for this assignment is to design a linear classifier from scratch only using numpy and autograd. (No tensorflow!)
* Build a linear classifer. The loss function is designed by cross-entropy and softmax functions.
*  Details can be found in [Tensorflow tutorial](https://www.tensorflow.org/tutorials/mnist/beginners/).

####[Assignment2 result]####
* (100 random batch / 1000 training / 0.5 learning rate) = 86~89% accuracy
* (200 order batch / 1000 training / 0.5 learning rate) = 90.35% accuracy
* cost function plot 해보니 계속 감소하는게 아니라 noise가 있었다. 그래서 random batch의 경우 70% accuracy가 나오기도 했다.
