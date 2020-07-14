# Authors: Decebal Constantin Mocanu et al.;
# Code associated with SCADS Summer School 2020 tutorial "	Scalable Deep Learning Tutorial"; https://www.scads.de/de/summerschool2020
# This is a pre-alpha free software and was tested in Windows 10 with Python 3.7.6, Numpy 1.17.2, SciPy 1.4.1, Numba 0.48.0

# If you use parts of this code please cite the following article:
# @article{Mocanu2018SET,
#   author  =    {Mocanu, Decebal Constantin and Mocanu, Elena and Stone, Peter and Nguyen, Phuong H. and Gibescu, Madeleine and Liotta, Antonio},
#   journal =    {Nature Communications},
#   title   =    {Scalable Training of Artificial Neural Networks with Adaptive Sparse Connectivity inspired by Network Science},
#   year    =    {2018},
#   doi     =    {10.1038/s41467-018-04316-3}
# }

# If you have space please consider citing also these articles

# @phdthesis{Mocanu2017PhDthesis,
#   title     =    "Network computations in artificial intelligence",
#   author    =    "D.C. Mocanu",
#   year      =    "2017",
#   isbn      =    "978-90-386-4305-2",
#   publisher =    "Eindhoven University of Technology",
# }

# @article{Liu2019onemillion,
#   author  =    {Liu, Shiwei and Mocanu, Decebal Constantin and Mocanu and Ramapuram Matavalam, Amarsagar Reddy and Pei, Yulong Pei and Pechenizkiy, Mykola},
#   journal =    {arXiv:1901.09181},
#   title   =    {Sparse evolutionary Deep Learning with over one million artificial neurons on commodity hardware},
#   year    =    {2019},
# }

# We thank to:
# Thomas Hagebols: for performing a thorough analyze on the performance of SciPy sparse matrix operations
# Ritchie Vink (https://www.ritchievink.com): for making available on Github a nice Python implementation of fully connected MLPs. This SET-MLP implementation was built on top of his MLP code:
#                                             https://github.com/ritchie46/vanilla-machine-learning/blob/master/vanilla_mlp.py

from keras.losses import *
from scipy.sparse import lil_matrix
from scipy.sparse import coo_matrix
from scipy.sparse import dok_matrix
from utils.nn_functions import *
from numba import njit, prange
import datetime
import os
import numpy as np
import sys

stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
sys.stderr = stderr


@njit(parallel=True, fastmath=True, cache=True)
def backpropagation_updates_numpy(a, delta, rows, cols, out):
    for i in prange(out.shape[0]):
        s = 0
        for j in range(a.shape[0]):
            s += a[j, rows[i]] * delta[j, cols[i]]
        out[i] = s / a.shape[0]


@njit(fastmath=True)
def find_first_pos(array, value):
    idx = (np.abs(array - value)).argmin()
    return idx


@njit(fastmath=True)
def find_last_pos(array, value):
    idx = (np.abs(array - value))[::-1].argmin()
    return array.shape[0] - idx


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


def create_sparse_weights(epsilon, n_rows, n_cols):
    # He uniform initialization
    limit = np.sqrt(6. / float(n_rows))

    mask_weights = np.random.rand(n_rows, n_cols)
    prob = 1 - (epsilon * (n_rows + n_cols)) / (n_rows * n_cols)  # normal to have 8x connections

    # generate an Erdos Renyi sparse weights mask
    weights = lil_matrix((n_rows, n_cols))
    n_params = np.count_nonzero(mask_weights[mask_weights >= prob])
    weights[mask_weights >= prob] = np.random.uniform(-limit, limit, n_params)
    # print("Create sparse matrix with ", weights.getnnz(), " connections and ",
    #       (weights.getnnz() / (n_rows * n_cols)) * 100, "% density level")
    weights = weights.tocsr()
    return weights


def array_intersect(a, b):
    # this are for array intersection
    n_rows, n_cols = a.shape
    dtype = {'names': ['f{}'.format(i) for i in range(n_cols)], 'formats': n_cols * [a.dtype]}
    return np.in1d(a.view(dtype), b.view(dtype))  # boolean return


class SET_MLP:
    def __init__(self, dimensions, activations, **config):
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

        self.epsilon = config['epsilon']  # control the sparsity level as discussed in the paper
        self.zeta = config['zeta']  # the fraction of the weights removed
        self.dropout_rate = config['dropout_rate']  # dropout rate
        self.dimensions = dimensions
        self.batch_size = config['batch_size']

        self.save_filename = ""
        self.input_layer_connections = []

        # Weights and biases are initiated by index. For a one hidden layer net you will have a w[1] and w[2]
        self.w = {}
        self.b = {}
        self.pdw = {}
        self.pdd = {}
        self.activations = {}

        for i in range(len(dimensions) - 1):
            self.w[i + 1] = create_sparse_weights(self.epsilon, dimensions[i],
                                                dimensions[i + 1])  # create sparse weight matrices
            self.b[i + 1] = np.zeros(dimensions[i + 1], dtype='float32')
            self.activations[i + 2] = activations[i]

        # limit = np.sqrt(6. / float(dimensions[-2]))
        # self.w[len(dimensions) - 1] = csr_matrix(np.random.uniform(-limit, limit,
        #                                                            (dimensions[-2], dimensions[-1])), dtype='float32')
        # self.b[len(dimensions) - 1] = np.zeros(dimensions[-1], dtype='float32')
        # self.activations[len(dimensions)] = activations[-1]

        if config['loss'] == 'mse':
            self.loss = MSE(self.activations[self.n_layers])
        elif config['loss'] == 'cross_entropy':
            self.loss = CrossEntropy()
        else:
            raise NotImplementedError("The given loss function is  ot implemented")

    def parameters(self):
        """
                Retrieve the network parameters.
                :return: model parameters.
        """

        params = {
            'w': self.w,
            'b': self.b,
            'pdw': self.pdw,
            'pdd': self.pdd,
        }

        return params

    def set_parameters(self, params):
        self.w = params['w']
        self.b = params['b']
        self.pdw = params['pdw']
        self.pdd = params['pdd']

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
        dw = coo_matrix(self.w[self.n_layers - 1], dtype='float32')
        # compute backpropagation updates
        backpropagation_updates_numpy(a[self.n_layers - 1], delta, dw.row, dw.col, dw.data)

        update_params = {
            self.n_layers - 1: (dw.tocsr(),  np.mean(delta, axis=0))
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

            dw = coo_matrix(self.w[i - 1], dtype='float32')

            # compute backpropagation updates
            backpropagation_updates_numpy(a[i - 1], delta, dw.row, dw.col, dw.data)

            update_params[i - 1] = (dw.tocsr(),  np.mean(delta, axis=0))

        return update_params

    def train_on_batch(self, x, y):
        z, a, masks = self._feed_forward(x, True)
        return self._back_prop(z, a, masks, y)

    def test_on_batch(self, x, y):
        accuracy, activations = self.predict(x, y)
        return self.loss.loss(y, activations), accuracy

    def get_core_input_connections(self):
        values = np.sort(self.w[1].data)
        first_zero_pos = find_first_pos(values, 0)
        last_zero_pos = find_last_pos(values, 0)

        largest_negative = values[int((1 - self.zeta) * first_zero_pos)]
        smallest_positive = values[
            int(min(values.shape[0] - 1, last_zero_pos + self.zeta * (values.shape[0] - last_zero_pos)))]

        wlil = self.w[1].tolil()
        wdok = dok_matrix((self.dimensions[0], self.dimensions[1]), dtype="float32")

        # remove the weights closest to zero
        keep_connections = 0
        for ik, (row, data) in enumerate(zip(wlil.rows, wlil.data)):
            for jk, val in zip(row, data):
                if (val < largest_negative) or (val > smallest_positive):
                    wdok[ik, jk] = val
                    keep_connections += 1
        return wdok.tocsr().getnnz(axis=1)

    def weights_evolution_I(self):
        # this represents the core of the SET procedure. It removes the weights closest to zero in each layer and add new random weights
        for i in range(1, self.n_layers - 1):

            values = np.sort(self.w[i].data)
            first_zero_pos = find_first_pos(values, 0)
            last_zero_pos = find_last_pos(values, 0)

            largest_negative = values[int((1 - self.zeta) * first_zero_pos)]
            smallest_positive = values[
                int(min(values.shape[0] - 1, last_zero_pos + self.zeta * (values.shape[0] - last_zero_pos)))]

            wlil = self.w[i].tolil()
            pdwlil = self.pdw[i].tolil()
            wdok = dok_matrix((self.dimensions[i - 1], self.dimensions[i]), dtype="float32")
            pdwdok = dok_matrix((self.dimensions[i - 1], self.dimensions[i]), dtype="float32")

            # remove the weights closest to zero
            keep_connections = 0
            for ik, (row, data) in enumerate(zip(wlil.rows, wlil.data)):
                for jk, val in zip(row, data):
                    if (val < largest_negative) or (val > smallest_positive):
                        wdok[ik, jk] = val
                        pdwdok[ik, jk] = pdwlil[ik, jk]
                        keep_connections += 1
            limit = np.sqrt(6. / float(self.dimensions[i]))

            # add new random connections
            for kk in range(self.w[i].data.shape[0] - keep_connections):
                ik = np.random.randint(0, self.dimensions[i - 1])
                jk = np.random.randint(0, self.dimensions[i])
                while (wdok[ik, jk] != 0):
                    ik = np.random.randint(0, self.dimensions[i - 1])
                    jk = np.random.randint(0, self.dimensions[i])
                wdok[ik, jk] = np.random.uniform(-limit, limit)
                pdwdok[ik, jk] = 0

            self.pdw[i] = pdwdok.tocsr()
            self.w[i] = wdok.tocsr()

    def weights_evolution_II(self):
        # this represents the core of the SET procedure. It removes the weights closest to zero in each layer and add new random weights
        # improved running time using numpy routines - Amarsagar Reddy Ramapuram Matavalam (amar@iastate.edu)
        for i in range(1, self.n_layers - 1):
            # uncomment line below to stop evolution of dense weights more than 80% non-zeros
            # if self.w[i].count_nonzero() / (self.w[i].get_shape()[0]*self.w[i].get_shape()[1]) < 0.8:
                t_ev_1 = datetime.datetime.now()
                # converting to COO form - Added by Amar
                wcoo = self.w[i].tocoo()
                vals_w = wcoo.data
                rows_w = wcoo.row
                cols_w = wcoo.col

                pdcoo = self.pdw[i].tocoo()
                vals_pd = pdcoo.data
                rows_pd = pdcoo.row
                cols_pd = pdcoo.col
                # print("Number of non zeros in W and PD matrix before evolution in layer",i,[np.size(valsW), np.size(valsPD)])
                values = np.sort(self.w[i].data)
                first_zero_pos = find_first_pos(values, 0)
                last_zero_pos = find_last_pos(values, 0)

                largest_negative = values[int((1-self.zeta) * first_zero_pos)]
                smallest_positive = values[int(min(values.shape[0] - 1, last_zero_pos + self.zeta * (values.shape[0] - last_zero_pos)))]

                #remove the weights (W) closest to zero and modify PD as well
                vals_w_new = vals_w[(vals_w > smallest_positive) | (vals_w < largest_negative)]
                rows_w_new = rows_w[(vals_w > smallest_positive) | (vals_w < largest_negative)]
                cols_w_new = cols_w[(vals_w > smallest_positive) | (vals_w < largest_negative)]

                new_w_row_col_index = np.stack((rows_w_new, cols_w_new), axis=-1)
                old_pd_row_col_index = np.stack((rows_pd, cols_pd), axis=-1)

                new_pd_row_col_index_flag = array_intersect(old_pd_row_col_index, new_w_row_col_index)  # careful about order

                vals_pd_new = vals_pd[new_pd_row_col_index_flag]
                rows_pd_new = rows_pd[new_pd_row_col_index_flag]
                cols_pd_new = cols_pd[new_pd_row_col_index_flag]

                self.pdw[i] = coo_matrix((vals_pd_new, (rows_pd_new, cols_pd_new)), (self.dimensions[i - 1], self.dimensions[i])).tocsr()

                # if i == 1:
                #     self.input_layer_connections.append(coo_matrix((vals_w_new, (rows_w_new, cols_w_new)),
                #                                                    (self.dimensions[i - 1], self.dimensions[i])).getnnz(axis=1))
                #     np.savez_compressed(self.save_filename + "_input_connections.npz",
                #                         inputLayerConnections=self.input_layer_connections)

                # add new random connections
                keep_connections = np.size(rows_w_new)
                length_random = vals_w.shape[0] - keep_connections
                limit = np.sqrt(6. / float(self.dimensions[i - 1]))
                random_vals = np.random.uniform(-limit, limit, length_random)
                zero_vals = 0*random_vals  # explicit zeros

                # adding  (wdok[ik,jk]!=0): condition
                while length_random > 0:
                    ik = np.random.randint(0, self.dimensions[i - 1], size=length_random, dtype='int32')
                    jk = np.random.randint(0, self.dimensions[i], size=length_random, dtype='int32')

                    random_w_row_col_index = np.stack((ik, jk), axis=-1)
                    random_w_row_col_index = np.unique(random_w_row_col_index, axis=0)  # removing duplicates in new rows&cols
                    oldW_row_col_index = np.stack((rows_w_new, cols_w_new), axis=-1)

                    unique_flag = ~array_intersect(random_w_row_col_index, oldW_row_col_index)  # careful about order & tilda

                    ik_new = random_w_row_col_index[unique_flag][:,0]
                    jk_new = random_w_row_col_index[unique_flag][:,1]
                    # be careful - row size and col size needs to be verified
                    rows_w_new = np.append(rows_w_new, ik_new)
                    cols_w_new = np.append(cols_w_new, jk_new)

                    length_random = vals_w.shape[0]-np.size(rows_w_new) # this will constantly reduce lengthRandom

                # adding all the values along with corresponding row and column indices - Added by Amar
                vals_w_new = np.append(vals_w_new, random_vals) # be careful - we can add to an existing link ?
                # vals_pd_new = np.append(vals_pd_new, zero_vals) # be careful - adding explicit zeros - any reason??
                if vals_w_new.shape[0] != rows_w_new.shape[0]:
                    print("not good")
                self.w[i] = coo_matrix((vals_w_new, (rows_w_new, cols_w_new)), (self.dimensions[i-1], self.dimensions[i])).tocsr()

                # print("Number of non zeros in W and PD matrix after evolution in layer",i,[(self.w[i].data.shape[0]), (self.pdw[i].data.shape[0])])

                t_ev_2 = datetime.datetime.now()
                # print("Weights evolution time for layer", i, "is", t_ev_2 - t_ev_1)

    def weights_evolution_III(self):
        # this represents the core of the SET procedure. It removes the weights closest to zero in each layer and add new random weights
        #evolve all layers, except the one from the last hidden layer to the output layer
        input_layer_connections = []
        for i in range(1, self.n_layers - 1):
            # uncomment line below to stop evolution of dense weights more than 80% non-zeros
            #if(self.w[i].count_nonzero()/(self.w[i].get_shape()[0]*self.w[i].get_shape()[1]) < 0.8):
                # converting to COO form
                wcoo = self.w[i].tocoo()
                vals_w = wcoo.data
                rows_w = wcoo.row
                cols_w = wcoo.col

                # print("Number of non zeros in W and PD matrix before evolution in layer",i,[np.size(vals_w), np.size(valsPD)])
                values = np.sort(self.w[i].data)
                first_zero_pos = find_first_pos(values, 0)
                last_zero_pos = find_last_pos(values, 0)

                largest_negative = values[int((1 - self.zeta) * first_zero_pos)]
                smallest_positive = values[
                    int(min(values.shape[0] - 1, last_zero_pos + self.zeta * (values.shape[0] - last_zero_pos)))]

                # remove the weights (W) closest to zero and modify PD as well
                vals_w_new = vals_w[(vals_w > smallest_positive) | (vals_w < largest_negative)]
                rows_w_new = rows_w[(vals_w > smallest_positive) | (vals_w < largest_negative)]
                cols_w_new = cols_w[(vals_w > smallest_positive) | (vals_w < largest_negative)]

                # if i == 1:
                #     input_layer_connections.append(coo_matrix((vals_w_new, (rows_w_new, cols_w_new)),
                #                        (self.dimensions[i - 1], self.dimensions[i])).getnnz(axis=1))

                # add new random connections
                keep_connections = np.size(rows_w_new)
                length_random = vals_w.shape[0] - keep_connections
                limit = np.sqrt(6. / float(self.dimensions[i]))
                random_vals = np.random.uniform(-limit, limit, length_random)

                # adding  (wdok[ik,jk]!=0): condition
                while length_random > 0:
                    ik = np.random.randint(0, self.dimensions[i - 1], size=length_random, dtype='int32')
                    jk = np.random.randint(0, self.dimensions[i], size=length_random, dtype='int32')

                    random_w_row_col_index = np.stack((ik, jk), axis=-1)
                    random_w_row_col_index = np.unique(random_w_row_col_index, axis=0)  # removing duplicates in new rows&cols
                    old_w_row_col_index = np.stack((rows_w_new, cols_w_new), axis=-1)

                    unique_flag = ~array_intersect(random_w_row_col_index, old_w_row_col_index)  # careful about order & tilda

                    ik_new = random_w_row_col_index[unique_flag][:, 0]
                    jk_new = random_w_row_col_index[unique_flag][:, 1]
                    # be careful - row size and col size needs to be verified
                    rows_w_new = np.append(rows_w_new, ik_new)
                    cols_w_new = np.append(cols_w_new, jk_new)

                    length_random = vals_w.shape[0] - np.size(rows_w_new)  # this will constantly reduce length_random

                # adding all the values along with corresponding row and column indices
                vals_w_new = np.append(vals_w_new, random_vals)

                if vals_w_new.shape[0] != rows_w_new.shape[0]:
                    print("not good")
                self.w[i] = coo_matrix((vals_w_new, (rows_w_new, cols_w_new)),
                                       (self.dimensions[i - 1], self.dimensions[i])).tocsr()

                # print("Number of non zeros in W and PD matrix after evolution in layer",i,[(self.w[i].data.shape[0]), (self.pdw[i].data.shape[0])])

        return input_layer_connections

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
