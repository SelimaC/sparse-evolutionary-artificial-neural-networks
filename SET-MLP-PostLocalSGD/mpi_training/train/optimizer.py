### Optimizers used to update master process weights

import numpy as np
from numba import njit
import logging
import scipy.sparse as sparse
from scipy.sparse import coo_matrix

class Optimizer(object):
    """Base class for optimization algorithms.
        Currently doesn't do anything."""

    def __init__(self):
        pass

    def reset(self):
        pass

    def apply_update(self, weights, gradient):
        raise NotImplementedError


class VanillaSGD(Optimizer):
    """Stochastic gradient descent with no extra frills.
          learning_rate: learning rate parameter for SGD"""

    def __init__(self, lr):
        super(VanillaSGD, self).__init__()
        self.learning_rate = lr

    def apply_update(self, weights, gradient):
        """Move weights in the direction of the gradient, by the amount of the
            learning rate."""

        for index, v in gradient.items():
            dw = v[0]
            delta = v[1]

            dw = retain_valid_updates(weights['w'][index], dw)

            weights['pdw'][index] = - self.learning_rate * dw
            weights['pdd'][index] = - self.learning_rate * delta

            weights['w'][index] += weights['pdw'][index]
            weights['b'][index] += weights['pdd'][index]

        return weights


class MomentumSGD(Optimizer):
    """Stochastic gradient descent with momentum and weight decay
          learning_rate: learning rate parameter for SGD"""

    def __init__(self, lr, weight_decay, momentum, n_workers):
        super(MomentumSGD, self).__init__()
        self.learning_rate = lr
        self.base_lr = 0.01
        self.epoch = 0
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.lr_decay = 0.1
        self.milestones = [5, 100, 175]
        self.current_milestone = 0
        self.n_workers = n_workers

    def apply_update(self, weights, gradient, epoch=0, nesterov=True):
        """Move weights in the direction of the gradient, by the amount of the
            learning rate."""

        self.epoch = epoch

        # Leaning rate scheduler
        # if self.epoch <= 5:  # Gradually warmup phase
        #     old_lr = self.learning_rate
        #     self.learning_rate = self.base_lr * ((self.n_workers - 1.0) * self.epoch / 5 + 1.0)
        #     self.momentum *= (self.learning_rate / old_lr)
        #     self.momentum = min(0.99, self.momentum)

        if self.epoch >= 100:  # First decay
            old_lr = self.learning_rate
            self.learning_rate *= self.lr_decay
            # self.momentum *= (self.learning_rate / old_lr)
            # self.momentum = min(0.99, self.momentum)

        if self.epoch >= 150:  # Second decay
            self.learning_rate *= self.lr_decay
            old_lr = self.learning_rate
            # self.momentum *= (self.learning_rate / old_lr)
            # self.momentum = min(0.99, self.momentum)
        # logging.info(f"Epoch {self.epoch}, learning rate {self.learning_rate}, momentum {self.momentum}")
        for index, v in gradient.items():
            dw = v[0]
            delta = v[1]

            dw = retain_valid_updates(weights['w'][index], dw)

            # perform the update with momentum
            if index not in weights['pdw']:
                weights['pdw'][index] = - self.learning_rate * dw
                weights['pdd'][index] = - self.learning_rate * delta
            else:
                weights['pdw'][index] = self.momentum * weights['pdw'][index] - self.learning_rate * dw
                weights['pdd'][index] = self.momentum * weights['pdd'][index] - self.learning_rate * delta

            if nesterov:
                weights['w'][index] += self.momentum * weights['pdw'][index] - self.learning_rate * dw - self.weight_decay * weights['w'][index]
                weights['b'][index] += self.momentum * weights['pdd'][index] - self.learning_rate * delta - self.weight_decay * weights['b'][index]
            else:
                weights['w'][index] += weights['pdw'][index] - self.weight_decay * weights['w'][index]
                weights['b'][index] += weights['pdd'][index] - self.weight_decay * weights['b'][index]

        return weights


class GEM(Optimizer):
    """GEM optimizer
        learning_rate: base learning rate, kept constant
        momentum: momentum term, constant
        kappa: Proxy amplification. Experimental results show 2 is a good value.
        """

    def __init__(self, learning_rate=0.01, momentum=0.9, kappa=1.0):
        super(GEM, self).__init__()
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.kappa = kappa
        self.epsilon = 1e-16

        self.central_variable_moment = {'w': {}, 'b': {}}
        self.stale = {'w': {}, 'b': {}}
        self.moment = {'pdw': {}, 'pdd': {}}

    def begin_compute_update(self, gradient):

        for index, v in gradient.items():
            dw = v[0]
            delta = v[1]

            if index not in self.moment['pdw']:
                self.moment['pdw'][index] = - self.learning_rate * dw
                self.moment['pdd'][index] = - self.learning_rate * delta
            else:
                self.moment['pdw'][index] = self.momentum * self.moment['pdw'][index] - self.learning_rate * dw
                self.moment['pdd'][index] = self.momentum * self.moment['pdd'][index] - self.learning_rate * delta

    def gradient_energy_matching(self, gradient):
        update_gem = {}

        for idx, v in gradient.items():
            dw = - self.learning_rate * v[0]
            delta = - self.learning_rate * v[1]

            proxy = self.kappa * np.abs(self.moment['pdw'][idx])

            central_variable = np.abs(self.central_variable_moment['w'][idx])

            update = np.abs(dw)
            pi_w = sparse_divide_nonzero(proxy - central_variable, update)
            pi_w.data[np.isnan(pi_w.data)] = self.epsilon
            np.clip(pi_w.data, 0., 5., out=pi_w.data)  # For numerical stability.

            proxy = self.kappa * np.abs(self.moment['pdd'][idx])
            central_variable = np.abs(self.central_variable_moment['b'][idx])
            update = np.abs(delta)
            pi_b = (proxy - central_variable) / (update + self.epsilon)
            np.clip(pi_b, 0., 5., out=pi_b)  # For numerical stability.

            update_gem[idx] = pi_w.multiply(dw), pi_b * delta

        return update_gem

    def compute_update(self, weights, gradient):

        for idx, b in weights['b'].items():
            if idx in self.stale['b']:
                self.central_variable_moment['b'][idx] = (b - self.stale['b'][idx])
            else:
                self.central_variable_moment['b'][idx] = np.zeros_like(b)
            self.stale['b'][idx] = np.copy(b)

        for idx, w in weights['w'].items():
            if idx in self.stale['w']:
                self.central_variable_moment['w'][idx] = (w - self.stale['w'][idx])
            else:
                self.central_variable_moment['w'][idx] = sparse.csr_matrix(w.shape, dtype='float64')
            self.stale['w'][idx] = w.copy()

        update = self.gradient_energy_matching(gradient)

        return update

    def apply_update(self, weights, gradient):
        """Add the update to the weights."""

        for index, v in gradient.items():
            dw = v[0]
            delta = v[1]

            # perform the update with momentum
            if index not in weights['pdw']:
                weights['pdw'][index] = dw
                weights['pdd'][index] = delta
            else:
                weights['pdw'][index] = self.momentum * weights['pdw'][index] + dw
                weights['pdd'][index] = self.momentum * weights['pdd'][index] + delta

            weights['w'][index] += weights['pdw'][index]  # - self.weight_decay * weights['w'][index]
            weights['b'][index] += weights['pdd'][index]  # - self.weight_decay * weights['b'][index]

        return weights


def sparse_divide_nonzero(a, b):
    inv_b = b.copy()
    inv_b.data = 1 / (inv_b.data + 1e-16)
    return a.multiply(inv_b)


def get_optimizer(name):
    """Get optimizer class by string identifier"""
    lookup = {
            # Native optimizers
            'sgd':           VanillaSGD,
            'sgdm':          MomentumSGD,
            'gem':           GEM,
            }
    return lookup[name]

def array_intersect(A, B):
    # this are for array intersection
    nrows, ncols = A.shape
    dtype = {'names': ['f{}'.format(i) for i in range(ncols)], 'formats': ncols * [A.dtype]}
    return np.in1d(A.view(dtype), B.view(dtype), assume_unique=True)  # boolean return


def retain_valid_updates_II(weights, gradient):
    weights = weights.tocoo()
    gradient = gradient.tocoo()

    valsPD, rowsPD, colsPD = gradient.data, gradient.row, gradient.col

    weights_indices = np.stack((weights.row, weights.col), axis=-1)
    gradient_indices = np.stack((rowsPD, colsPD), axis=-1)
    indices = array_intersect(gradient_indices, weights_indices)

    valsPDNew = valsPD[indices]
    rowsPDNew = rowsPD[indices]
    colsPDNew = colsPD[indices]

    gradient = coo_matrix((valsPDNew, (rowsPDNew, colsPDNew)), gradient.shape).tocsr()

    return gradient


def retain_valid_updates(weights, gradient):
    cols = gradient.shape[1]
    weights = weights.tocoo()
    gradient = gradient.tocoo()
    K_weights = np.array(weights.row * cols + weights.col)
    K_gradient = np.array(gradient.row * cols + gradient.col)

    indices = np.setdiff1d(K_gradient, K_weights, assume_unique=True)
    if len(indices) != 0:
        rows, cols = np.unravel_index(indices, gradient.shape)
        gradient = gradient.tocsr()
        gradient[rows, cols] = 0
        gradient.eliminate_zeros()

    return gradient


def retain_valid_weights(correct_weights, new_weights):
    cols = new_weights.shape[1]
    correct_weights = correct_weights.tocoo()
    new_weights = new_weights.tocoo()

    K_correct_weights = np.array(correct_weights.row * cols + correct_weights.col)
    K_new_weights = np.array(new_weights.row * cols + new_weights.col)

    indices = np.setdiff1d(K_new_weights, K_correct_weights, assume_unique=True)
    if len(indices) != 0:
        rows, cols = np.unravel_index(indices, new_weights.shape)
        correct_weights = correct_weights.tolil()
        correct_weights[rows, cols] = new_weights.tocsr()[rows, cols]

    return correct_weights.tocsr()


class OptimizerBuilder(object):
    """Builds a  optimizer"""

    def __init__(self, name, config=None):
        self.name = name
        self.config = config
        if self.config is None:
            self.config = {}
        if self.name == 'sgd' and 'lr' not in self.config:
            logging.warning("Learning rate for SGD not set, using 0.1")
            self.config['lr'] = 0.1

    def build(self):
        from keras.optimizers import deserialize
        opt_config = {'class_name': self.name, 'config': self.config}
        opt = deserialize(opt_config)
        return opt