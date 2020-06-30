# Author: Decebal Constantin Mocanu et al.;
# Proof of concept implementation of Sparse Evolutionary Training (SET) of Multi Layer Perceptron (MLP) on CIFAR10 using Keras and a mask over weights.
# This implementation can be used to test SET in varying conditions, using the Keras framework versatility, e.g. various optimizers, activation layers, tensorflow
# Also it can be easily adapted for Convolutional Neural Networks or other models which have dense layers
# However, due the fact that the weights are stored in the standard Keras format (dense matrices), this implementation can not scale properly.
# If you would like to build and SET-MLP with over 100000 neurons, please use the pure Python implementation from the folder "SET-MLP-Sparse-Python-Data-Structures"

# This is a pre-alpha free software and was tested with Python 3.5.2, Keras 2.1.3, Keras_Contrib 0.0.2, Tensorflow 1.5.0, Numpy 1.14;
# The code is distributed in the hope that it may be useful, but WITHOUT ANY WARRANTIES; The use of this software is entirely at the user's own risk;
# For an easy understanding of the code functionality please read the following articles.

# If you use parts of this code please cite the following articles:
#@article{Mocanu2018SET,
#  author =        {Mocanu, Decebal Constantin and Mocanu, Elena and Stone, Peter and Nguyen, Phuong H. and Gibescu, Madeleine and Liotta, Antonio},
#  journal =       {Nature Communications},
#  title =         {Scalable Training of Artificial Neural Networks with Adaptive Sparse Connectivity inspired by Network Science},
#  year =          {2018},
#  doi =           {10.1038/s41467-018-04316-3}
#}

#@Article{Mocanu2016XBM,
#author="Mocanu, Decebal Constantin and Mocanu, Elena and Nguyen, Phuong H. and Gibescu, Madeleine and Liotta, Antonio",
#title="A topological insight into restricted Boltzmann machines",
#journal="Machine Learning",
#year="2016",
#volume="104",
#number="2",
#pages="243--270",
#doi="10.1007/s10994-016-5570-z",
#url="https://doi.org/10.1007/s10994-016-5570-z"
#}

#@phdthesis{Mocanu2017PhDthesis,
#title = "Network computations in artificial intelligence",
#author = "D.C. Mocanu",
#year = "2017",
#isbn = "978-90-386-4305-2",
#publisher = "Eindhoven University of Technology",
#}

from __future__ import division
from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras import optimizers
import numpy as np
from keras import backend as K
#Please note that in newer versions of keras_contrib you may encounter some import errors. You can find a fix for it on the Internet, or as an alternative you can try other activations functions.
from keras_contrib.layers.advanced_activations.srelu import SReLU
from keras.datasets import mnist
from keras.utils import np_utils
import urllib.request as urllib2
import pandas as pd

class Constraint(object):

    def __call__(self, w):
        return w

    def get_config(self):
        return {}

class MaskWeights(Constraint):

    def __init__(self, mask):
        self.mask = mask
        self.mask = K.cast(self.mask, K.floatx())

    def __call__(self, w):
        w *= self.mask
        return w

    def get_config(self):
        return {'mask': self.mask}


def find_first_pos(array, value):
    idx = (np.abs(array - value)).argmin()
    return idx


def find_last_pos(array, value):
    idx = (np.abs(array - value))[::-1].argmin()
    return array.shape[0] - idx


def createWeightsMask(epsilon,noRows, noCols):
    # generate an Erdos Renyi sparse weights mask
    mask_weights = np.random.rand(noRows, noCols)
    prob = 1 - (epsilon * (noRows + noCols)) / (noRows * noCols)  # normal tp have 8x connections
    mask_weights[mask_weights < prob] = 0
    mask_weights[mask_weights >= prob] = 1
    noParameters = np.sum(mask_weights)
    print ("Create Sparse Matrix: No parameters, NoRows, NoCols ",noParameters,noRows,noCols)
    return [noParameters,mask_weights]


class SET_MLP_MADELON:
    def __init__(self):
        # set model parameters
        self.epsilon = 13 # control the sparsity level as discussed in the paper
        self.zeta = 0.3 # the fraction of the weights removed
        self.batch_size = 100 # batch size
        self.maxepoches = 1000 # number of epochs
        self.learning_rate = 0.1 # SGD learning rate
        self.num_classes = 2 # number of classes
        self.momentum=0.9 # SGD momentum

        # generate an Erdos Renyi sparse weights mask for each layer
        [self.noPar1, self.wm1] = createWeightsMask(self.epsilon,500, 1000)
        [self.noPar2, self.wm2] = createWeightsMask(self.epsilon,1000, 500)
        [self.noPar3, self.wm3] = createWeightsMask(self.epsilon,500, 1000)

        # initialize layers weights
        self.w1 = None
        self.w2 = None
        self.w3 = None
        self.w4 = None

        # initialize weights for SReLu activation function
        self.wSRelu1 = None
        self.wSRelu2 = None
        self.wSRelu3 = None

        # create a SET-MLP model
        self.create_model()

        # train the SET-MLP model
        self.train()


    def create_model(self):

        # create a SET-MLP model for CIFAR10 with 3 hidden layers
        self.model = Sequential()
        self.model.add(Dense(1000, name="sparse_1",kernel_constraint=MaskWeights(self.wm1),weights=self.w1, input_shape=(500,)))
        self.model.add(SReLU(name="srelu1",weights=self.wSRelu1))
        self.model.add(Dropout(0.3))
        self.model.add(Dense(500, name="sparse_2",kernel_constraint=MaskWeights(self.wm2),weights=self.w2))
        self.model.add(SReLU(name="srelu2",weights=self.wSRelu2))
        self.model.add(Dropout(0.3))
        self.model.add(Dense(1000, name="sparse_3",kernel_constraint=MaskWeights(self.wm3),weights=self.w3))
        self.model.add(SReLU(name="srelu3",weights=self.wSRelu3))
        self.model.add(Dropout(0.3))
        self.model.add(Dense(1,  name="dense_4",activation='sigmoid'))


    def rewireMask_basic(self, weights, noWeights):
        # rewire weight matrix

        # remove zeta largest negative and smallest positive weights
        values = np.sort(weights.ravel())
        firstZeroPos = find_first_pos(values, 0)
        lastZeroPos = find_last_pos(values, 0)
        largestNegative = values[int((1-self.zeta) * firstZeroPos)]
        smallestPositive = values[int(min(values.shape[0] - 1, lastZeroPos +self.zeta * (values.shape[0] - lastZeroPos)))]
        rewiredWeights = weights.copy();
        rewiredWeights[rewiredWeights > smallestPositive] = 1;
        rewiredWeights[rewiredWeights < largestNegative] = 1;
        rewiredWeights[rewiredWeights != 1] = 0;
        weightMaskCore = rewiredWeights.copy()

        # add zeta random weights
        nrAdd = 0
        noRewires = noWeights - np.sum(rewiredWeights)
        while (nrAdd < noRewires):
            i = np.random.randint(0, rewiredWeights.shape[0])
            j = np.random.randint(0, rewiredWeights.shape[1])
            if (rewiredWeights[i, j] == 0):
                rewiredWeights[i, j] = 1
                nrAdd += 1

        return [rewiredWeights, weightMaskCore]

    def rewireMask(self,weights, outgoing_weights, noWeights):
        # rewire weight matrix
        sum_incoming_weights = np.abs(weights).sum(axis=0)
        sum_outgoing_weights = np.abs(outgoing_weights).sum(axis=1)
        edges = sum_incoming_weights + sum_outgoing_weights

        t = np.percentile(edges, 20)
        edges = np.where(edges <= t, 0, edges)
        ids = np.argwhere(edges == 0)

        rewiredWeights = weights.copy();
        rewiredWeights[:, ids] = 0;
        rewiredWeights[rewiredWeights != 0] = 1;
        weightMaskCore = rewiredWeights.copy()

        # add zeta random weights
        nrAdd = 0
        noRewires = noWeights - np.sum(rewiredWeights)
        while (nrAdd < noRewires):
            i = np.random.randint(0, rewiredWeights.shape[0])
            j = np.random.randint(0, rewiredWeights.shape[1])
            if (rewiredWeights[i, j] == 0):
                rewiredWeights[i, j] = 1
                nrAdd += 1

        return [rewiredWeights, weightMaskCore]

    def weightsEvolution(self, epoch):
        # this represents the core of the SET procedure. It removes the weights closest to zero in each layer and add new random weights
        self.w1 = self.model.get_layer("sparse_1").get_weights()
        self.w2 = self.model.get_layer("sparse_2").get_weights()
        self.w3 = self.model.get_layer("sparse_3").get_weights()
        self.w4 = self.model.get_layer("dense_4").get_weights()

        self.wSRelu1 = self.model.get_layer("srelu1").get_weights()
        self.wSRelu2 = self.model.get_layer("srelu2").get_weights()
        self.wSRelu3 = self.model.get_layer("srelu3").get_weights()


        if False:#epoch > 250 and epoch % 100 == 0:
            [self.wm1, self.wm1Core] = self.rewireMask(self.w1[0], self.w2[0], self.noPar1)
            [self.wm2, self.wm2Core] = self.rewireMask(self.w2[0], self.w3[0], self.noPar2)
            [self.wm3, self.wm3Core] = self.rewireMask(self.w3[0], self.w4[0], self.noPar3)
        else:
            [self.wm1, self.wm1Core] = self.rewireMask_basic(self.w1[0], self.noPar1)
            [self.wm2, self.wm2Core] = self.rewireMask_basic(self.w2[0], self.noPar2)
            [self.wm3, self.wm3Core] = self.rewireMask_basic(self.w3[0], self.noPar3)

        self.w1[0] = self.w1[0] * self.wm1Core
        self.w2[0] = self.w2[0] * self.wm2Core
        self.w3[0] = self.w3[0] * self.wm3Core

    def train(self):

        # read Madelon data
        [x_train,x_test,y_train,y_test]=self.read_data()

        self.model.summary()

        # training process in a for loop
        self.accuracies_per_epoch=[]
        for epoch in range(0, self.maxepoches):
            sgd = optimizers.SGD(momentum=0.9, learning_rate=0.05)
            self.model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
            # Shuffle the data
            seed = np.arange(x_train.shape[0])
            np.random.shuffle(seed)
            x_ = x_train[seed]
            y_ = y_train[seed]

            for j in range(x_train.shape[0] // self.batch_size):
                k = j * self.batch_size
                l = (j + 1) * self.batch_size

                historytemp = self.model.train_on_batch(x=x_[k:l], y=y_[k:l])

            print("\nSET-MLP Epoch ", epoch)
            result_test = self.model.evaluate(x=x_test, y=y_test, verbose=0)
            print("Metrics test: ", result_test)
            result_train = self.model.evaluate(x=x_train, y=y_train, verbose=0)
            print("Metrics train: ", result_train)
            self.accuracies_per_epoch.append((result_train[0], result_train[1],
                                              result_test[0], result_test[1]))

            #ugly hack to avoid tensorflow memory increase for multiple fit_generator calls. Theano shall work more nicely this but it is outdated in general
            self.weightsEvolution(epoch)
            K.clear_session()
            self.create_model()
            np.savetxt("SReluWeights1_madelon.txt", self.wSRelu1)
            np.savetxt("SReluWeights2_madelon.txt", self.wSRelu2)
            np.savetxt("SReluWeights3_madelon.txt", self.wSRelu3)
        self.model.save_weights('madelon_weights_fulltraining.h5')

        self.accuracies_per_epoch=np.asarray(self.accuracies_per_epoch)

    def read_data(self):

        # Also makes it so if the URL changes I can change it in one place easily
        train_data_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/madelon/MADELON/madelon_train.data'
        val_data_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/madelon/MADELON/madelon_valid.data'
        train_resp_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/madelon/MADELON/madelon_train.labels'
        val_resp_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/madelon/madelon_valid.labels'
        test_data_url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/madelon/MADELON/madelon_test.data'

        # This data isn't standard csv so using numpy to load it as text
        # Download the data straight from the web
        x_train = np.loadtxt(urllib2.urlopen(train_data_url))
        y_train = np.loadtxt(urllib2.urlopen(train_resp_url))
        x_val = np.loadtxt(urllib2.urlopen(val_data_url))
        y_val = np.loadtxt(urllib2.urlopen(val_resp_url))
        x_test = np.loadtxt(urllib2.urlopen(test_data_url))

        y_train = np.where(y_train == -1, 0, 1)
        y_val = np.where(y_val == -1, 0, 1)

        # scale the data in 0..1
        x_min = x_train.min()
        x_max = x_train.max()
        x_train = (x_train - x_min) / (x_max - x_min)
        x_test = (x_test - x_min) / (x_max - x_min)
        x_val = (x_val - x_min) / (x_max - x_min)

        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_val = x_val.astype('float32')

        return [x_train, x_val, y_train, y_val]


if __name__ == '__main__':

    # create and run a SET-MLP model on CIFAR10
    model=SET_MLP_MADELON()

    # save accuracies over for all training epochs
    # in "results" folder you can find the output of running this file
    np.savetxt("results/set_new_version_mlp_srelu_sgd_madelon_acc.txt", np.asarray(model.accuracies_per_epoch))




