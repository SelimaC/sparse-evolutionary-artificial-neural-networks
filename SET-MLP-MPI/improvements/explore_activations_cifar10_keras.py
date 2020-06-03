import matplotlib
#matplotlib.use('Agg')
from scipy import stats
from scipy.sparse import csr_matrix, find
import matplotlib.pyplot as plt
import datetime
import networkx as nx
from networkx.algorithms import bipartite
import numpy as np
from models.set_mlp_sequential import *
from utils.load_data import *
from utils.load_data import *
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, ReLU
from keras import optimizers
import numpy as np
import time
from models.sgdw_keras import SGDW
import datetime
from keras import backend as K
#Please note that in newer versions of keras_contrib you may encounter some import errors. You can find a fix for it on the Internet, or as an alternative you can try other activations functions.
from keras_contrib.layers.advanced_activations.srelu import SReLU
from keras.datasets import cifar10
from keras.utils import np_utils
import argparse
import keras
X_train, Y_train, X_test, Y_test = load_cifar10_data_not_flattened(50000, 10000)

config = {
            'n_processes': 3,
            'n_epochs': 10,
            'batch_size': 100,
            'dropout_rate': 0.3,
            'seed': 0,
            'lr': 0.01,
            'lr_decay': 0.0,
            'zeta': 0.3,
            'epsilon': 20,
            'momentum': 0.9,
            'weight_decay': 0.0,
            'n_hidden_neurons': 1000,
            'n_training_samples': 60000,
            'n_testing_samples': 10000,
            'loss': 'cross_entropy'
        }


def createWeightsMask(epsilon, n_rows, n_cols):
    # generate an Erdos Renyi sparse weights mask
    mask_weights = np.random.rand(n_rows, n_cols)
    prob = 1 - (epsilon * (n_rows + n_cols)) / (n_rows * n_cols)  # normal tp have 8x connections
    mask_weights[mask_weights < prob] = 0
    mask_weights[mask_weights >= prob] = 1
    no_parameters = np.sum(mask_weights)
    print ("Create Sparse Matrix: No parameters, NoRows, NoCols ", no_parameters, n_rows, n_cols)
    return [no_parameters, mask_weights]


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

# initialize layers weights
w1 = None
w2 = None
w3 = None
w4 = None
# initialize weights for SReLu activation function
wSRelu1 = None
wSRelu2 = None
wSRelu3 = None

[noPar1, wm1] = createWeightsMask(20,32*32*3, 4000)
[noPar2, wm2] = createWeightsMask(20,4000, 1000)
[noPar3, wm3] = createWeightsMask(20,1000, 4000)

model = Sequential()
model.add(Flatten(input_shape=(32, 32, 3)))
model.add(Dense(4000, name="sparse_1",kernel_constraint=MaskWeights(wm1),weights=w1))
model.add(SReLU(name="srelu1",weights=wSRelu1))
model.add(Dropout(0.3))
model.add(Dense(1000, name="sparse_2",kernel_constraint=MaskWeights(wm2),weights=w2))
model.add(SReLU(name="srelu2",weights=wSRelu2))
model.add(Dropout(0.3))
model.add(Dense(4000, name="sparse_3",kernel_constraint=MaskWeights(wm3),weights=w3))
model.add(SReLU(name="srelu3",weights=wSRelu3))
model.add(Dropout(0.3))
model.add(Dense(10, name="dense_4", weights=w4)) #please note that there is no need for a sparse output layer as the number of classes is much smaller than the number of input hidden neurons
model.add(Activation('softmax'))

model.load_weights('my_model_weights_fulltraining.h5')

sgd = optimizers.SGD(momentum=0.9, learning_rate=0.01)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

result_test = model.model.evaluate(x=X_test, y=Y_test, verbose=0)
print("Metrics test before pruning: ", result_test)
result_train = model.model.evaluate(x=X_train, y=Y_train, verbose=0)
print("Metrics train before pruning: ", result_train)

weights = {}
weights[1] = model.get_layer("sparse_1").get_weights()[0]
weights[2] = model.get_layer("sparse_2").get_weights()[0]
weights[3] = model.get_layer("sparse_3").get_weights()[0]
weights[4] = model.get_layer("dense_4").get_weights()[0]
b ={}
b[1] = model.get_layer("sparse_1").get_weights()[1]
b[2] = model.get_layer("sparse_2").get_weights()[1]
b[3] = model.get_layer("sparse_3").get_weights()[1]
b[4] = model.get_layer("dense_4").get_weights()[1]

print("\nNon zero before pruning: ")
for k, w in weights.items():
    print(np.count_nonzero(w))


wSRelu1 = model.get_layer("srelu1").get_weights()
wSRelu2 = model.get_layer("srelu2").get_weights()
wSRelu3 = model.get_layer("srelu3").get_weights()

for k, w in weights.items():
    w_sparse = csr_matrix(w)
    i, j, v = find(w_sparse)
    # plt.hist(np.round(v,2), bins=100)
    # plt.title(f'Weight distribution layer {k}')
    # plt.xlabel("value")
    # plt.ylabel("Frequency")
    # plt.show()

    # positive_std = np.std(weights[weights > 0])
    # zscore_pos = stats.zscore(v)
    #
    # plt.plot(zscore_pos)
    # plt.show()
    w=w_sparse.toarray()
    negative_mean = np.median(w[w < 0])
    positive_mean = np.median(w[w > 0])

    p95 = np.percentile(v, 95)
    p75 = np.percentile(v, 75)
    p50 = np.percentile(v, 50)
    p25 = np.percentile(v, 25)
    p5 = np.percentile(v, 5)
    p20 = np.percentile(v, 20)
    p80 = np.percentile(v, 80)

    # negative_std = np.std(weights[weights < 0])
    # zscore_neg = stats.zscore(weights[weights < 0])
    eps = 0.05
    # w[(w < positive_mean) & (w > 0)] = 0.0
    # w[(w > negative_mean - eps)  & (w < 0)] = 0.0
    # w[(w <= np.round(p75,  2)) & (w > 0)] = 0.0
    # w[(w >= np.round(p25,  2)) & (w < 0)] = 0.0
    w[np.abs(w) <= p50] = 0.0
    # w[(np.abs(np.round(w, 2)) == np.round(positive_mean, 2)) | (np.abs(np.round(w, 2)) == np.round(negative_mean, 2))] = 0.0
    # weights[(np.round(weights, 2) != np.round(p5, 2)) & (np.round(weights, 2) != np.round(p25, 2)) &
    #         (np.round(weights, 2) != np.round(p50, 2)) & (np.round(weights, 2) != np.round(p75, 2)) & (np.round(weights, 2) != np.round(p95, 2))] = 0.0
    # weights[np.round(weights, 2) == np.round(p5,  2)] = 0.0
    w[np.round(w, 2) == np.round(p25, 2)] = 0.0
    w[np.round(w, 2) == np.round(p75, 2)] = 0.0
    #weights[np.round(weights, 2) == np.round(p95, 2)] = 0.0
    w = csr_matrix(w)
    i, j, v = find(w)
    # plt.hist(v, bins=100)
    # plt.title(f'Weight distribution layer {k}')
    # plt.xlabel("value")
    # plt.ylabel("Frequency")
    # plt.show()
    weights[k] = w.toarray()

print("\nNon zero after pruning: ")
for k, w in weights.items():
    print(np.count_nonzero(w))

model = Sequential()
model.add(Flatten(input_shape=(32, 32, 3)))
model.add(Dense(4000, name="sparse_1",kernel_constraint=MaskWeights(wm1),weights=w1))
model.add(SReLU(name="srelu1",weights=wSRelu1))
model.add(Dropout(0.3))
model.add(Dense(1000, name="sparse_2",kernel_constraint=MaskWeights(wm2),weights=w2))
model.add(SReLU(name="srelu2",weights=wSRelu2))
model.add(Dropout(0.3))
model.add(Dense(4000, name="sparse_3",kernel_constraint=MaskWeights(wm3),weights=w3))
model.add(SReLU(name="srelu3",weights=wSRelu3))
model.add(Dropout(0.3))
model.add(Dense(10, name="dense_4", weights=w4)) #please note that there is no need for a sparse output layer as the number of classes is much smaller than the number of input hidden neurons
model.add(Activation('softmax'))
sgd = optimizers.SGD(momentum=0.9, learning_rate=0.01)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
model.load_weights('my_model_weights_fulltraining.h5')
model.get_layer("sparse_1").set_weights([weights[1], b[1]])
model.get_layer("sparse_2").set_weights([weights[2], b[2]])
model.get_layer("sparse_3").set_weights([weights[3], b[3]])
model.get_layer("dense_4").set_weights([weights[4], b[4]])

result_test = model.model.evaluate(x=X_test, y=Y_test, verbose=0)
print("Metrics test after pruning: ", result_test)
result_train = model.model.evaluate(x=X_train, y=Y_train, verbose=0)
print("Metrics train after pruning: ", result_train)