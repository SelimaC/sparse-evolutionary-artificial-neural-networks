# Authors: Decebal Constantin Mocanu et al.;
# Code associated with SCADS Summer School 2020 tutorial "	Scalable Deep Learning Tutorial"; https://www.scads.de/de/summerschool2020
# This is a pre-alpha free software and was tested in Windows 10 with Python 3.7.6, Numpy 1.17.2, SciPy 1.4.1, Numba 0.48.0

# If you use parts of this code please cite the following article:
#@article{Mocanu2018SET,
#  author =        {Mocanu, Decebal Constantin and Mocanu, Elena and Stone, Peter and Nguyen, Phuong H. and Gibescu, Madeleine and Liotta, Antonio},
#  journal =       {Nature Communications},
#  title =         {Scalable Training of Artificial Neural Networks with Adaptive Sparse Connectivity inspired by Network Science},
#  year =          {2018},
#  doi =           {10.1038/s41467-018-04316-3}
#}

# If you have space please consider citing also these articles

#@phdthesis{Mocanu2017PhDthesis,
#title = "Network computations in artificial intelligence",
#author = "D.C. Mocanu",
#year = "2017",
#isbn = "978-90-386-4305-2",
#publisher = "Eindhoven University of Technology",
#}

#@article{Liu2019onemillion,
#  author =        {Liu, Shiwei and Mocanu, Decebal Constantin and Mocanu and Ramapuram Matavalam, Amarsagar Reddy and Pei, Yulong Pei and Pechenizkiy, Mykola},
#  journal =       {arXiv:1901.09181},
#  title =         {Sparse evolutionary Deep Learning with over one million artificial neurons on commodity hardware},
#  year =          {2019},
#}

# We thank to:
# Thomas Hagebols: for performing a thorough analyze on the performance of SciPy sparse matrix operations
# Ritchie Vink (https://www.ritchievink.com): for making available on Github a nice Python implementation of fully connected MLPs. This SET-MLP implementation was built on top of his MLP code:
#                                             https://github.com/ritchie46/vanilla-machine-learning/blob/master/vanilla_mlp.py


import numpy as np
from numba import njit
import datetime
from nn_functions import *


@njit(fastmath=True, cache=True)
def compute_accuracy(activations, y_test):
    correct_classification = 0
    for j in range(y_test.shape[0]):
        if np.argmax(activations[j]) == np.argmax(y_test[j]):
            correct_classification += 1
    return correct_classification / y_test.shape[0]


@njit(fastmath=True, cache=True)
def dropout(x, rate):
    noise_shape = x.shape
    noise = np.random.uniform(0., 1., noise_shape)
    keep_prob = 1. - rate
    scale = np.float32(1 / keep_prob)
    keep_mask = noise >= rate
    return x * scale * keep_mask, keep_mask


class Dense_MLP:
    def __init__(self, dimensions, activations):
        """
        :param dimensions: (tpl/ list) Dimensions of the neural net. (input, hidden layer, output)
        :param activations: (tpl/ list) Activations functions.

        Example of three hidden layer with
        - 3312 input features
        - 3000 hidden neurons
        - 3000 hidden neurons
        - 3000 hidden neurons
        - 5 output classes


        layers -->    [1,        2,     3,     4,     5]
        ----------------------------------------

        dimensions =  (3312,     3000,  3000,  3000,  5)
        activations = (          Relu,  Relu,  Relu,  Sigmoid)
        """
        self.n_layers = len(dimensions)
        self.loss = None
        self.learning_rate = None
        self.momentum = None
        self.weight_decay = None
        self.dropout_rate = 0  # dropout rate
        self.dimensions = dimensions

        # Weights and biases are initiated by index. For a one hidden layer net you will have a w[1] and w[2]
        self.w = {}
        self.b = {}
        self.pdw = {}
        self.pdd = {}

        # Activations are also initiated by index. For the example we will have activations[2] and activations[3]
        self.activations = {}
        for i in range(len(dimensions) - 1):
            # He uniform initialization
            limit = np.sqrt(6. / float(dimensions[i]))
            self.w[i + 1] = np.random.uniform(-limit, limit, (dimensions[i], dimensions[i + 1]))
            self.b[i + 1] = np.zeros(dimensions[i + 1])
            self.activations[i + 2] = activations[i]

    def _feed_forward(self, x, drop=False):
        """
        Execute a forward feed through the network.
        :param x: (array) Batch of input data vectors.
        :return: (tpl) Node outputs and activations per layer. The numbering of the output is equivalent to the layer numbers.
        """
        # w(x) + b
        z = {}

        # activations: f(z)
        a = {1: x}  # First layer has no activations as input. The input x is the input.
        masks = {}

        for i in range(1, self.n_layers):
            z[i + 1] = a[i] @ self.w[i] + self.b[i]
            a[i + 1] = self.activations[i + 1].activation(z[i + 1])
            if drop:
                if i < self.n_layers - 1:
                    # apply dropout
                    a[i + 1], keep_mask = dropout(a[i + 1], self.dropout_rate)
                    masks[i + 1] = keep_mask

        return z, a, masks

    def _back_prop(self, z, a, masks, y_true):
        """
        The input dicts keys represent the layers of the net.

        a = { 1: x,
              2: f(w1(x) + b1)
              3: f(w2(a2) + b2)
              4: f(w3(a3) + b3)
              5: f(w4(a4) + b4)
              }

        :param z: (dict) w(x) + b
        :param a: (dict) f(z)
        :param y_true: (array) One hot encoded truth vector.
        :return:
        """
        keep_prob = 1.
        if self.dropout_rate > 0:
            keep_prob = np.float32(1. - self.dropout_rate)

        # Determine partial derivative and delta for the output layer.
        # delta output layer
        delta = self.loss.delta(y_true, a[self.n_layers])
        dw = np.dot(a[self.n_layers - 1].T, delta)

        update_params = {
            self.n_layers - 1: (dw, np.mean(delta, axis=0))
        }

        # In case of three layer net will iterate over i = 2 and i = 1
        # Determine partial derivative and delta for the rest of the layers.
        # Each iteration requires the delta from the previous layer, propagating backwards.
        for i in reversed(range(2, self.n_layers)):
            # dropout for the backpropagation step
            if keep_prob != 1:
                delta = (delta @ self.w[i].transpose()) * self.activations[i].prime(z[i])
                delta = delta * masks[i]
                delta /= keep_prob
            else:
                delta = (delta @ self.w[i].transpose()) * self.activations[i].prime(z[i])

            dw = np.dot(a[i - 1].T, delta)

            update_params[i - 1] = (dw, np.mean(delta, axis=0))
        for k, v in update_params.items():
            self._update_w_b(k, v[0], v[1])

    def _update_w_b(self, index, dw, delta):
        """
        Update weights and biases.

        :param index: (int) Number of the layer
        :param dw: (array) Partial derivatives
        :param delta: (array) Delta error.
        """

        # perform the update with momentum
        if (index not in self.pdw):
            self.pdw[index] = - self.learning_rate * dw
            self.pdd[index] = - self.learning_rate * delta
        else:
            self.pdw[index] = self.momentum * self.pdw[index] - self.learning_rate * dw
            self.pdd[index] = self.momentum * self.pdd[index] - self.learning_rate * delta

        self.w[index] += self.pdw[index] - self.weight_decay * self.w[index]
        self.b[index] += self.pdd[index] - self.weight_decay * self.b[index]

    def fit(self, x, y_true, x_test, y_test, loss, epochs, batch_size, learning_rate=1e-3, momentum=0.9,
            weight_decay=0.0002, zeta=0.3, dropoutrate=0, testing=True, save_filename=""):
        """
        :param x: (array) Containing parameters
        :param y_true: (array) Containing one hot encoded labels.
        :return (array) A 2D array of metrics (epochs, 3).
        """
        if not x.shape[0] == y_true.shape[0]:
            raise ValueError("Length of x and y arrays don't match")

        # Initiate the loss object with the final activation function
        self.loss = loss()
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.zeta = zeta
        self.droprate = dropoutrate
        self.save_filename = save_filename

        maximum_accuracy = 0

        metrics = np.zeros((epochs, 4))

        for i in range(epochs):
            # Shuffle the data
            seed = np.arange(x.shape[0])
            np.random.shuffle(seed)
            x_ = x[seed]
            y_ = y_true[seed]

            # training
            t1 = datetime.datetime.now()

            for j in range(x.shape[0] // batch_size):
                k = j * batch_size
                l = (j + 1) * batch_size
                z, a, masks = self._feed_forward(x_[k:l], True)

                self._back_prop(z, a, masks, y_[k:l])

            t2 = datetime.datetime.now()

            print("\nFC-MLP Epoch ", i)
            print("Training time: ", t2 - t1)

            # test model performance on the test data at each epoch
            # this part is useful to understand model performance and can be commented for production settings
            if (testing):
                t3 = datetime.datetime.now()
                accuracy_test, activations_test = self.predict(x_test, y_test, batch_size)
                accuracy_train, activations_train = self.predict(x, y_true, batch_size)
                t4 = datetime.datetime.now()
                maximum_accuracy = max(maximum_accuracy, accuracy_test)
                loss_test = self.loss.loss(y_test, activations_test)
                loss_train = self.loss.loss(y_true, activations_train)
                metrics[i, 0] = loss_train
                metrics[i, 1] = loss_test
                metrics[i, 2] = accuracy_train
                metrics[i, 3] = accuracy_test
                print("Testing time: ", t4 - t3,"; Loss train: ", loss_train, "; Loss test: ", loss_test, "; Accuracy train: ", accuracy_train,"; Accuracy test: ", accuracy_test,
                      "; Maximum accuracy test: ", maximum_accuracy)

                      # save performance metrics values in a file
            if (save_filename != ""):
                np.savetxt(save_filename, metrics)

        return metrics

    def predict(self, x_test, y_test, batch_size=100):
        """
        :param x_test: (array) Test input
        :param y_test: (array) Correct test output
        :param batch_size:
        :return: (flt) Classification accuracy
        :return: (array) A 2D array of shape (n_cases, n_classes).
        """
        activations = np.zeros((y_test.shape[0], y_test.shape[1]))
        for j in range(x_test.shape[0] // batch_size):
            k = j * batch_size
            l = (j + 1) * batch_size
            _, a_test, _ = self._feed_forward(x_test[k:l], drop=False)
            activations[k:l] = a_test[self.n_layers]
        accuracy = compute_accuracy(activations, y_test)
        return accuracy, activations

def load_fashion_mnist_data(noTrainingSamples,noTestingSamples):
    np.random.seed(0)

    data=np.load("data/fashion_mnist.npz")

    indexTrain=np.arange(data["X_train"].shape[0])
    np.random.shuffle(indexTrain)

    indexTest=np.arange(data["X_test"].shape[0])
    np.random.shuffle(indexTest)

    X_train=data["X_train"][indexTrain[0:noTrainingSamples],:]
    Y_train=data["Y_train"][indexTrain[0:noTrainingSamples],:]
    X_test=data["X_test"][indexTest[0:noTestingSamples],:]
    Y_test=data["Y_test"][indexTest[0:noTestingSamples],:]

    #normalize in 0..1
    X_train = X_train.astype('float64') / 255.
    X_test = X_test.astype('float64') / 255.

    return X_train,Y_train,X_test,Y_test

if __name__ == "__main__":

    for i in [5]:
        #load data
        noTrainingSamples = 10000 #max 60000 for Fashion MNIST
        noTestingSamples = 5000  # max 10000 for Fshion MNIST
        X_train, Y_train, X_test, Y_test = load_fashion_mnist_data(noTrainingSamples,noTestingSamples)

        #set model parameters
        noHiddenNeuronsLayer = 1000
        noTrainingEpochs = 500
        batchSize = 50
        dropoutRate = 0.2
        learningRate = 0.001
        momentum=0.9
        weightDecay=0.0002

        np.random.seed(i)

        # create FC-MLP ( fully-connected MLP)
        dense_mlp = Dense_MLP((X_train.shape[1], noHiddenNeuronsLayer, noHiddenNeuronsLayer, noHiddenNeuronsLayer, Y_train.shape[1]), (Relu, Relu, Relu, Softmax))

        # train FC-MLP
        dense_mlp.fit(X_train, Y_train, X_test, Y_test, loss=CrossEntropy, epochs=noTrainingEpochs, batch_size=batchSize, learning_rate=learningRate,
                      momentum=momentum, weight_decay=weightDecay, dropoutrate=dropoutRate, testing=True,
                      save_filename="Results/dense_mlp_"+str(noTrainingSamples)+"_training_samples_rand"+str(i))

        # test FC-MLP
        accuracy, _ = dense_mlp.predict(X_test, Y_test, batch_size=1)

        print("\nAccuracy of the last epoch on the testing data: ", accuracy)
