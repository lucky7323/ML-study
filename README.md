# ML-study
2016 winter ~ 2017 spring study with python

###[Assignment2 - Linear Classifier]###
__description__
* The goal of this assignment is to make a simple linear classifer for MNIST, which contains 60,000 handwritten digit images with 10 classes.
* The most important thing for this assignment is to design a linear classifier from scratch only using numpy and autograd. (No tensorflow!)
* Build a linear classifer. The loss function is designed by cross-entropy and softmax functions.
* Details can be found in [Tensorflow tutorial](https://www.tensorflow.org/tutorials/mnist/beginners/).

__result__
* (100 random batch / 1000 training / 0.5 learning rate) = 89~92% accuracy
* data preprocessing 전에는 80후반 accuracy, 후에는 92%까지 오름.
* Weight, bias initialize는 accuracy에 큰 영향이 없었음.

- - - -

###[Assignment3 - Neural Network]###
__description__
* The goal of this assingment is to implement a very simple feedforward neural network with numpy and autograd.
* The number of hidden layer and hidden units are not fixed. You can design your own network architecture.
* Dropout should be used for regularization.

__result__
* Weight, bias의 initialize를 0~1의 random값으로 했을 때 학습이 제대로 되지 않았다.
* Weight, bias를 mean=0, std=0.1의 normal distribution으로 initialize하니 85%정도 까지 올랐다.
* data preprocessing을 해주고 learning rate도 0.5로 올리는 등 parameter를 좀 변경하니 96%까지 accuracy가 올랐다.
