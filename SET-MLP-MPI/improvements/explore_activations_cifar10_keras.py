import matplotlib
#matplotlib.use('Agg')
import seaborn as sns
from scipy import stats
from sklearn.metrics import mean_squared_error, r2_score
from scipy.sparse import csr_matrix, find
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import datetime
from sklearn.metrics import classification_report
import networkx as nx
from networkx.algorithms import bipartite
import numpy as np
from scipy.stats import norm
from models.set_mlp_sequential import *
from utils.load_data import *
from keras import backend as K
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
# Force Keras to use CPU
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
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

model.load_weights('cifar10_weights_fulltraining.h5')

sgd = optimizers.SGD(momentum=0.9, learning_rate=0.01)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

result_test = model.model.evaluate(x=X_test, y=Y_test, verbose=0)
print("Metrics test before pruning: ", result_test)
result_train = model.model.evaluate(x=X_train, y=Y_train, verbose=0)
print("Metrics train before pruning: ", result_train)

Y_true = np.argmax(Y_test, axis=1) # Convert one-hot to index
y_pred = model.predict_classes(X_test)
print(classification_report(Y_true, y_pred))

matrix = confusion_matrix(Y_true, y_pred)
print(matrix.diagonal()/matrix.sum(axis=1))

incorrects = np.nonzero(y_pred.reshape((-1,)) != Y_true)[0]
incorrect_classes = y_pred[incorrects]
correct_classes = Y_true[incorrects]
for i in range(0, 10):
    count = np.argwhere(correct_classes==i).shape[0]
    print(f"Class {i} has {count} incorrect classifications")


weights = {}
weights[1] = model.get_layer("sparse_1").get_weights()[0]
np.savetxt("Weights1.txt", weights[1])
weights[2] = model.get_layer("sparse_2").get_weights()[0]
np.savetxt("Weights2.txt", weights[2])
weights[3] = model.get_layer("sparse_3").get_weights()[0]
np.savetxt("Weights3.txt", weights[3])
weights[4] = model.get_layer("dense_4").get_weights()[0]
np.savetxt("Weights4.txt", weights[4])
b ={}
b[1] = model.get_layer("sparse_1").get_weights()[1]
np.savetxt("Biases1.txt", b[1])
b[2] = model.get_layer("sparse_2").get_weights()[1]
np.savetxt("Biases2.txt", b[2])
b[3] = model.get_layer("sparse_3").get_weights()[1]
np.savetxt("Biases3.txt", b[3])
b[4] = model.get_layer("dense_4").get_weights()[1]
np.savetxt("Biases4.txt", b[4])
srelu_weights = {}
srelu_weights[1] = np.loadtxt("SReluWeights1_cifar10.txt")
srelu_weights[2] = np.loadtxt("SReluWeights2_cifar10.txt")
srelu_weights[3] = np.loadtxt("SReluWeights3_cifar10.txt")

def srelu(tl, al, tr, ar, x):
    if x >= tr:
        return tr + ar * (x - tr)
    if x <= tl:
        return tl + al * (x - tl)
    return x

vfun = np.vectorize(srelu)
x = np.linspace(-5, 5, 10000)

# model.get_layer("sparse_1").set_weights([weights[1], np.zeros(4000)])
# model.get_layer("sparse_2").set_weights([weights[2], np.zeros(1000)])
# model.get_layer("sparse_3").set_weights([weights[3], np.zeros(4000)])
# model.get_layer("dense_4").set_weights([weights[4], np.zeros(10)])
#
# result_test = model.model.evaluate(x=X_test, y=Y_test, verbose=0)
# print("Metrics test before pruning no bias: ", result_test)
# result_train = model.model.evaluate(x=X_train, y=Y_train, verbose=0)
# print("Metrics train before pruning no bias: ", result_train)

print("\nNon zero before pruning: ")
for k, w in weights.items():
    print(np.count_nonzero(w))


wSRelu1 = model.get_layer("srelu1").get_weights()
wSRelu2 = model.get_layer("srelu2").get_weights()
wSRelu3 = model.get_layer("srelu3").get_weights()
mean_activations = []
mean_inputs = []
for k, w in weights.items():
    if k==2:
        get_nth_layer_output = K.function([model.layers[0].input], [model.layers[4].output])
        inputs = get_nth_layer_output([X_test])[0]
        mean_input = inputs.mean(axis=0)

        get_nth_layer_output = K.function([model.layers[0].input], [model.layers[5].output])
        activations = get_nth_layer_output([X_test])[0]
        mean_activation = activations.mean(axis=0)
    if k==1:
        get_nth_layer_output = K.function([model.layers[0].input], [model.layers[1].output])
        inputs = get_nth_layer_output([X_test])[0]
        mean_input = inputs.mean(axis=0)

        get_nth_layer_output = K.function([model.layers[0].input], [model.layers[2].output])
        activations = get_nth_layer_output([X_test])[0]
        mean_activation = activations.mean(axis=0)
    if k==3:
        get_nth_layer_output = K.function([model.layers[0].input], [model.layers[7].output])
        inputs = get_nth_layer_output([X_test])[0]
        mean_input = inputs.mean(axis=0)

        get_nth_layer_output = K.function([model.layers[0].input], [model.layers[8].output])
        activations = get_nth_layer_output([X_test])[0]
        mean_activation = activations.mean(axis=0)
    if k==4:
        get_nth_layer_output = K.function([model.layers[0].input], [model.layers[10].output])
        inputs = get_nth_layer_output([X_test])[0]
        mean_input = inputs.mean(axis=0)

        get_nth_layer_output = K.function([model.layers[0].input], [model.layers[11].output])
        activations = get_nth_layer_output([X_test])[0]
        mean_activation = activations.mean(axis=0)

    # sns.distplot(mean_activation, hist=False, kde=True,
    #              kde_kws={'linewidth': 3},
    #              label='Layer ' + str(k))
    # plt.show()
    mean_activations.append(mean_activation)
    mean_inputs.append(mean_input)

    # sns.distplot(mean_input, hist=False, kde=True,
    #              kde_kws={'linewidth': 3},
    #              label='Input Layer ' + str(k))
    # sns.distplot(mean_activation, hist=False, kde=True,
    #              kde_kws={'linewidth': 3},
    #              label='Activation Layer ' + str(k))
    # plt.show()
    if k<=3:
        tl = srelu_weights[k][0]
        al = srelu_weights[k][1]
        tr = srelu_weights[k][2]
        ar = srelu_weights[k][3]

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

    unique, counts = np.unique(j, return_counts=True)
    incoming_edges = counts
    if k !=4:
        i, _, _ = find(weights[k+1])
        unique, counts = np.unique(i, return_counts=True)
        outgoing_edges = counts
        sum_incoming_weights = np.abs(weights[k]).sum(axis=0)
        sum_outgoing_weights = np.abs(weights[k+1]).sum(axis=1)  
        edges = sum_incoming_weights + sum_outgoing_weights
        connections = outgoing_edges + incoming_edges
    else:
        sum_incoming_weights = np.abs(weights[k]).sum(axis=0)
        edges = sum_incoming_weights
        connections = incoming_edges

    if k != 4:
        t_connections = np.percentile(connections, 25)
        t = np.percentile(edges, 50)
        t2 = np.percentile(edges, 100)
        idxs = (-edges).argsort()[:1000]
        print(
            f"Removing {edges[(edges<t) | (edges>t2)].shape[0]} neurons and {incoming_edges[(edges<t) | (edges>t2)].sum()} weights , weighted sum threshold is {t}, connection threshold is {t_connections}")
        edges = np.where((edges<t) | (edges>t2), 0, edges)
        ids = np.argwhere(edges==0)

        # if k==1:
        #     ids = np.argwhere(al>0.2)
        #     print(f"removing {len(ids)} neurons")
        # if k==2:
        #     ids = np.argwhere(al<0)
        #     print(f"removing {len(ids)} neurons")
        # if k==3:
        #     ids =np.argwhere(al>0.2)
        #     print(f"removing {len(ids)} neurons")

        #ids = list(set(np.concatenate((remove, ids), axis=None)))
        import matplotlib as mpl
        import pandas as pd

        mpl.rcParams['agg.path.chunksize'] = 10000

        # fig, axs = plt.subplots(2, sharex=True)
        # fig.suptitle(f'SReLu Layer {k} - {w.shape[1]} neurons')
        # for idx in idxs:
        #     y = vfun(tl[idx], al[idx], tr[idx], ar[idx], x)

            # df = pd.DataFrame()
            # df.input = inputs[idx]
            # df.activation=activations[idx]
            # df_mean = df.groupby('input')['activation'].mean()
            # axs[0].plot(x, y, '-')
            # axs[0].set_xlim([-5, 5])
            # sns.distplot(activations[idx], hist=False, kde=True,
            #              kde_kws={'linewidth': 3})
            # axs[1].scatter(inputs[idx], activations[idx].mean(), alpha=0.8, edgecolors='none', s=10)
            # axs[1].set_xlim([-5, 5])
            # axs[1].hist(activations[idx], bins=1000)
            # axs[1].set_xlim([-5, 5])

        #plt.show()
        #
        # for idx in idxs:
        #     ax2 = fig.add_subplot(2, 1, 2, sharex=ax1)
        #     # sns.distplot(activations[idx], hist=False, kde=True,
        #     #              kde_kws={'linewidth': 3})
        #     ax2.plot.hist(np.round(v, 2), bins=100)
        #plt.show()

        # Create plot
        #x, y = inputs.flatten(), activations.flatten()
        # plt.scatter(x, y, alpha=0.8, c='blue', edgecolors='none', s=30, label='Layer ' + str(k))
        # plt.title('Inputs VS Activations')
        # plt.legend(loc=2)
        # plt.show()

        # # Create linear regression object for positive inputs
        # regr_pos = LinearRegression()
        # x_pos_idxs = np.argwhere(x>=0)
        # regr_pos.fit(x[x_pos_idxs], y[x_pos_idxs])
        # # Make predictions using the testing set
        # y_pos_predict = regr_pos.predict(y[x_pos_idxs])
        # # The coefficients
        # print('Positive coefficients: \n', regr_pos.coef_)
        # # The mean squared error
        # print('Positive mean squared error: %.2f'
        #       % mean_squared_error(y[x_pos_idxs], y_pos_predict))
        # # The coefficient of determination: 1 is perfect prediction
        # print('Positive coefficient of determination: %.2f'
        #       % r2_score(y[x_pos_idxs], y_pos_predict))
        #
        # # Plot outputs
        # plt.plot(x[x_pos_idxs], y_pos_predict, color='blue', linewidth=3)
        # plt.show()
        #
        # # Create linear regression object for negative inputs
        # regr_neg = LinearRegression()
        # x_neg_idxs = np.argwhere(x < 0)
        # regr_neg.fit(x[x_neg_idxs], y[x_neg_idxs])
        # # Make predictions using the testing set
        # y_neg_predict = regr_neg.predict(y[x_neg_idxs])
        # # The coefficients
        # print('Negative coefficients: \n', regr_neg.coef_)
        # # The mean squaed error
        # print('Negative mean squared error: %.2f'
        #       % mean_squared_error(y[x_neg_idxs], y_neg_predict))
        # # The coefficient of determination: 1 is perfect prediction
        # print('Negative coefficient of determination: %.2f'
        #       % r2_score(y[x_neg_idxs], y_neg_predict))

        # Plot outputs
        # plt.plot(x[x_neg_idxs], y_neg_predict, color='blue', linewidth=3)
        # plt.show()

        # Prune weights
        w[:, ids] = 0
        if k == 3:
            weights[k+1][ids, :] = 0




    # negative_std = np.std(weights[weights < 0])
    # zscore_neg = stats.zscore(weights[weights < 0])
    #eps = 0.05
    # w[(w < positive_mean) & (w > 0)] = 0.0
    # w[(w > negative_mean - eps)  & (w < 0)] = 0.0
    # w[(w <= np.round(p75,  2)) & (w > 0)] = 0.0
    # w[(w >= np.round(p25,  2)) & (w < 0)] = 0.0
    #w[np.abs(w) <= p50] = 0.0
    # w[(np.abs(np.round(w, 2)) == np.round(positive_mean, 2)) | (np.abs(np.round(w, 2)) == np.round(negative_mean, 2))] = 0.0
    # weights[(np.round(weights, 2) != np.round(p5, 2)) & (np.round(weights, 2) != np.round(p25, 2)) &
    #         (np.round(weights, 2) != np.round(p50, 2)) & (np.round(weights, 2) != np.round(p75, 2)) & (np.round(weights, 2) != np.round(p95, 2))] = 0.0
    # weights[np.round(weights, 2) == np.round(p5,  2)] = 0.0
    # w[np.round(w, 3) == np.round(p25, 3)] = 0.0
    # w[np.round(w, 3) == np.round(p75, 3)] = 0.0
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

# Iterate through the activations
for idx, activations in enumerate(mean_activations):
    # Draw the density plot
    sns.distplot(activations, hist=False, kde=True,
                 kde_kws={'linewidth': 3},
                 label='Activations Layer ' + str(idx+1))
plt.show()

# Iterate through the pre-activations
for idx, input in enumerate(mean_inputs):
    # Draw the density plot
    sns.distplot(input, hist=False, kde=True,
                 kde_kws={'linewidth': 3},
                 label='Input Layer ' + str(idx+1))
plt.show()


np.savetxt("Weights1.txt", weights[1])
np.savetxt("Weights2.txt", weights[2])
np.savetxt("Weights3.txt", weights[3])
np.savetxt("Weights4.txt", weights[4])

np.savetxt("Biases1.txt", b[1])
np.savetxt("Biases2.txt", b[2])
np.savetxt("Biases3.txt", b[3])
np.savetxt("Biases4.txt", b[4])

wm1 = np.where(weights[1] > 0, 1,  0)
wm2 = np.where(weights[2] > 0, 1,  0)
wm3 = np.where(weights[3] > 0, 1,  0)
wm4 = np.where(weights[4] > 0, 1,  0)
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
#model.load_weights('cifar10_weights_fulltraining.h5')
model.get_layer("sparse_1").set_weights([weights[1], b[1]])
model.get_layer("sparse_2").set_weights([weights[2], b[2]])
model.get_layer("sparse_3").set_weights([weights[3], b[3]])
model.get_layer("dense_4").set_weights([weights[4], b[4]])

model.save_weights('my_model_weights_fulltraining_pruned.h5')

result_test = model.model.evaluate(x=X_test, y=Y_test, verbose=0)
print("Metrics test after pruning: ", result_test)
result_train = model.model.evaluate(x=X_train, y=Y_train, verbose=0)
print("Metrics train after pruning: ", result_train)
Y_true = np.argmax(Y_test, axis=1) # Convert one-hot to index
y_pred = model.predict_classes(X_test)
print(classification_report(Y_true, y_pred))
matrix = confusion_matrix(Y_true, y_pred)
print(matrix.diagonal()/matrix.sum(axis=1))
incorrects = np.nonzero(y_pred.reshape((-1,)) != Y_true)[0]
incorrect_classes = y_pred[incorrects]
correct_classes = Y_true[incorrects]
for i in range(0, 10):
    count = np.argwhere(correct_classes==i).shape[0]
    print(f"Class {i} has {count} incorrect classifications")
