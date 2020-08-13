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
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, ReLU
from keras import optimizers
import numpy as np
from keras import backend as K
#Please note that in newer versions of keras_contrib you may encounter some import errors. You can find a fix for it on the Internet, or as an alternative you can try other activations functions.
from keras_contrib.layers.advanced_activations.srelu import SReLU
from keras.datasets import fashion_mnist
from keras.utils import np_utils

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


class SET_MLP_FASHION_MNIST:
    def __init__(self):
        # set model parameters
        self.epsilon = 10 # control the sparsity level as discussed in the paper
        self.zeta = 0.3 # the fraction of the weights removed
        self.batch_size = 5 # batch size
        self.maxepoches = 1000 # number of epochs
        self.learning_rate = 0.005 # SGD learning rate
        self.num_classes = 18 # number of classes
        self.momentum=0.9 # SGD momentum

        # generate an Erdos Renyi sparse weights mask for each layer
        [self.noPar1, self.wm1] = createWeightsMask(self.epsilon,54675, 2750)
        [self.noPar2, self.wm2] = createWeightsMask(self.epsilon,2750, 2750)
        [self.noPar3, self.wm3] = createWeightsMask(self.epsilon,2750, 2750)

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
        self.model.add(Dense(2750, name="sparse_1",kernel_constraint=MaskWeights(self.wm1),weights=self.w1, input_shape=(54675,)))
        self.model.add(ReLU(name="srelu1",weights=self.wSRelu1))
        self.model.add(Dropout(0.3))
        self.model.add(Dense(2750, name="sparse_2",kernel_constraint=MaskWeights(self.wm2),weights=self.w2))
        self.model.add(ReLU(name="srelu2",weights=self.wSRelu2))
        self.model.add(Dropout(0.3))
        self.model.add(Dense(2750, name="sparse_3",kernel_constraint=MaskWeights(self.wm3),weights=self.w3))
        self.model.add(ReLU(name="srelu3",weights=self.wSRelu3))
        self.model.add(Dropout(0.3))
        self.model.add(Dense(self.num_classes, name="dense_4", weights=self.w4)) #please note that there is no need for a sparse output layer as the number of classes is much smaller than the number of input hidden neurons
        self.model.add(Activation('softmax'))


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

        # read CIFAR10 data
        [x_train,y_train,x_test,y_test]=self.read_data()

        self.model.summary()

        # training process in a for loop
        self.accuracies_per_epoch=[]
        for epoch in range(0, self.maxepoches):
            sgd = optimizers.SGD(momentum=0.9, learning_rate=0.005, clipvalue=0.5)
            self.model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
            # Shuffle the data
            seed = np.arange(x_train.shape[0])
            np.random.shuffle(seed)
            x_ = x_train[seed]
            y_ = y_train[seed]
            historytemp = None
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
            np.savetxt("SReluWeights1_fashionmnist.txt", self.wSRelu1)
            np.savetxt("SReluWeights2_fashionmnist.txt", self.wSRelu2)
            np.savetxt("SReluWeights3_fashionmnist.txt", self.wSRelu3)
        self.model.save_weights('fashionmnist_weights_fulltraining.h5')

        self.accuracies_per_epoch=np.asarray(self.accuracies_per_epoch)

    # The Leukemia dataset is obtained from the NCBI GEO repository with the accession number GSE13159
    def read_data(self):
        labels = np.loadtxt("../data/Leukemia/labels.txt")
        data = np.loadtxt("../data/Leukemia/values.txt")

        # Convert data into 'float32' type
        data = data.astype('float32')

        label_0 = np.argwhere(labels == 0)
        label_0_train_ids = label_0[0:26];
        label_0_test_ids = label_0[26:]

        label_1 = np.argwhere(labels == 1)
        label_1_train_ids = label_1[0:39];
        label_1_test_ids = label_1[39:]

        label_2 = np.argwhere(labels == 2)
        label_2_train_ids = label_2[0:24];
        label_2_test_ids = label_2[24:]

        label_3 = np.argwhere(labels == 3)
        label_3_train_ids = label_3[0:30];
        label_3_test_ids = label_3[30:]

        label_4 = np.argwhere(labels == 4)
        label_4_train_ids = label_4[0:19];
        label_4_test_ids = label_4[19:]

        label_5 = np.argwhere(labels == 5)
        label_5_train_ids = label_5[0:236];
        label_5_test_ids = label_5[236:]

        label_6 = np.argwhere(labels == 6)
        label_6_train_ids = label_6[0:25];
        label_6_test_ids = label_6[25:]

        label_7 = np.argwhere(labels == 7)
        label_7_train_ids = label_7[0:25];
        label_7_test_ids = label_7[25:]

        label_8 = np.argwhere(labels == 8)
        label_8_train_ids = label_8[0:26];
        label_8_test_ids = label_8[26:]

        label_9 = np.argwhere(labels == 9)
        label_9_train_ids = label_9[0:299];
        label_9_test_ids = label_9[299:]

        label_10 = np.argwhere(labels == 10)
        label_10_train_ids = label_10[0:51];
        label_10_test_ids = label_10[51:]

        label_11 = np.argwhere(labels == 11)
        label_11_train_ids = label_11[0:138];
        label_11_test_ids = label_11[138:]

        label_12 = np.argwhere(labels == 12)
        label_12_train_ids = label_12[0:48];
        label_12_test_ids = label_12[48:]

        label_13 = np.argwhere(labels == 13)
        label_13_train_ids = label_13[0:47];
        label_13_test_ids = label_13[47:]

        label_14 = np.argwhere(labels == 14)
        label_14_train_ids = label_14[0:116];
        label_14_test_ids = label_14[116:]

        label_15 = np.argwhere(labels == 15)
        label_15_train_ids = label_15[0:81];
        label_15_test_ids = label_15[81:]

        label_16 = np.argwhere(labels == 16)
        label_16_train_ids = label_16[0:158];
        label_16_test_ids = label_16[158:]

        label_17 = np.argwhere(labels == 17)
        label_17_train_ids = label_17[0:9];
        label_17_test_ids = label_17[9:]

        # index_data = np.arange(data.shape[0])
        # np.random.shuffle(index_data)
        #
        # data = data[index_data, :]
        # labels = labels[index_data]

        # replace nan with col means
        data = np.where(np.isnan(data), np.ma.array(data, mask=np.isnan(data)).mean(axis=0), data)

        labels = np.array(labels)
        data = np.array(data)

        x_test = np.vstack((data[label_0_test_ids.reshape(-1, )],
                            data[label_1_test_ids.reshape(-1, )],
                            data[label_2_test_ids.reshape(-1, )],
                            data[label_3_test_ids.reshape(-1, )],
                            data[label_4_test_ids.reshape(-1, )],
                            data[label_5_test_ids.reshape(-1, )],
                            data[label_6_test_ids.reshape(-1, )],
                            data[label_7_test_ids.reshape(-1, )],
                            data[label_8_test_ids.reshape(-1, )],
                            data[label_9_test_ids.reshape(-1, )],
                            data[label_10_test_ids.reshape(-1, )],
                            data[label_11_test_ids.reshape(-1, )],
                            data[label_12_test_ids.reshape(-1, )],
                            data[label_13_test_ids.reshape(-1, )],
                            data[label_14_test_ids.reshape(-1, )],
                            data[label_15_test_ids.reshape(-1, )],
                            data[label_16_test_ids.reshape(-1, )],
                            data[label_17_test_ids.reshape(-1, )]
                            ))
        y_test = np.hstack((labels[label_0_test_ids.reshape(-1, )],
                            labels[label_1_test_ids.reshape(-1, )],
                            labels[label_2_test_ids.reshape(-1, )],
                            labels[label_3_test_ids.reshape(-1, )],
                            labels[label_4_test_ids.reshape(-1, )],
                            labels[label_5_test_ids.reshape(-1, )],
                            labels[label_6_test_ids.reshape(-1, )],
                            labels[label_7_test_ids.reshape(-1, )],
                            labels[label_8_test_ids.reshape(-1, )],
                            labels[label_9_test_ids.reshape(-1, )],
                            labels[label_10_test_ids.reshape(-1, )],
                            labels[label_11_test_ids.reshape(-1, )],
                            labels[label_12_test_ids.reshape(-1, )],
                            labels[label_13_test_ids.reshape(-1, )],
                            labels[label_14_test_ids.reshape(-1, )],
                            labels[label_15_test_ids.reshape(-1, )],
                            labels[label_16_test_ids.reshape(-1, )],
                            labels[label_17_test_ids.reshape(-1, )]
                            ))
        x_train = np.vstack((data[label_0_train_ids.reshape(-1, )],
                             data[label_1_train_ids.reshape(-1, )],
                             data[label_2_train_ids.reshape(-1, )],
                             data[label_3_train_ids.reshape(-1, )],
                             data[label_4_train_ids.reshape(-1, )],
                             data[label_5_train_ids.reshape(-1, )],
                             data[label_6_train_ids.reshape(-1, )],
                             data[label_7_train_ids.reshape(-1, )],
                             data[label_8_train_ids.reshape(-1, )],
                             data[label_9_train_ids.reshape(-1, )],
                             data[label_10_train_ids.reshape(-1, )],
                             data[label_11_train_ids.reshape(-1, )],
                             data[label_12_train_ids.reshape(-1, )],
                             data[label_13_train_ids.reshape(-1, )],
                             data[label_14_train_ids.reshape(-1, )],
                             data[label_15_train_ids.reshape(-1, )],
                             data[label_16_train_ids.reshape(-1, )],
                             data[label_17_train_ids.reshape(-1, )]
                             ))
        y_train = np.hstack((labels[label_0_train_ids.reshape(-1, )],
                             labels[label_1_train_ids.reshape(-1, )],
                             labels[label_2_train_ids.reshape(-1, )],
                             labels[label_3_train_ids.reshape(-1, )],
                             labels[label_4_train_ids.reshape(-1, )],
                             labels[label_5_train_ids.reshape(-1, )],
                             labels[label_6_train_ids.reshape(-1, )],
                             labels[label_7_train_ids.reshape(-1, )],
                             labels[label_8_train_ids.reshape(-1, )],
                             labels[label_9_train_ids.reshape(-1, )],
                             labels[label_10_train_ids.reshape(-1, )],
                             labels[label_11_train_ids.reshape(-1, )],
                             labels[label_12_train_ids.reshape(-1, )],
                             labels[label_13_train_ids.reshape(-1, )],
                             labels[label_14_train_ids.reshape(-1, )],
                             labels[label_15_train_ids.reshape(-1, )],
                             labels[label_16_train_ids.reshape(-1, )],
                             labels[label_17_train_ids.reshape(-1, )]
                             ))

        # x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.33, random_state=42)
        y_train = np_utils.to_categorical(y_train, 18)
        y_test = np_utils.to_categorical(y_test, 18)

        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')

        return x_train, y_train, x_test, y_test


if __name__ == '__main__':

    # create and run a SET-MLP model on CIFAR10
    model=SET_MLP_FASHION_MNIST()

    # save accuracies over for all training epochs
    # in "results" folder you can find the output of running this file
    np.savetxt("results/set_mlp_srelu_sgd_leukemia_acc.txt", np.asarray(model.accuracies_per_epoch))




