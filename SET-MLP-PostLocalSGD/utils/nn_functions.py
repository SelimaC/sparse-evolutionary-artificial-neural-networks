import numpy as np
from scipy.stats import skew


class Relu:
    @staticmethod
    def activation(z):
        z[z < 0] = 0
        return z

    @staticmethod
    def prime(z):
        z[z < 0] = 0
        z[z > 0] = 1
        return z


class InverseRelu:
    @staticmethod
    def activation(z):
        z[z > 0] = 0
        return - z

    @staticmethod
    def prime(z):
        z[z > 0] = 0
        z[z < 0] = -1
        return z


class LeakyRelu:
    @staticmethod
    def activation(z):
        z[z < 0] *= 0.25
        return z

    @staticmethod
    def prime(z):
        z[z < 0] = 0.25
        z[z > 0] = 1
        return z


class LinearSigmoid:
    @staticmethod
    def activation(z):
        p25 = np.percentile(z, 25)
        p75 = np.percentile(z, 75)
        z[(z > p25) & (z < p75)] = 0
        return z

    @staticmethod
    def prime(z):
        p25 = np.percentile(z, 25)
        p75 = np.percentile(z, 75)
        z[(z <= p25) | (z >= p75)] = 1
        z[(z > p25) & (z < p75)] = 0
        return z


class Elu:
    @staticmethod
    def activation(z):
        z = np.where(z <= 0, (np.exp(z) - 1), z)
        return z

    @staticmethod
    def prime(z):
        z = np.where(z <= 0, Elu.activation(z) + 1, 1)
        return z


class RReLu:
    def __init__(self, n_neurons=None):
        self.n_neurons = n_neurons
        if n_neurons == 4000:
            self.left_slopes = np.random.uniform(-0.75, 0.75, self.n_neurons)
        else:
            self.left_slopes = np.random.uniform(-0.75, 0.75, self.n_neurons)

    def activation(self, z):
        for i in range(self.n_neurons):
            z[:, i] = np.where(z[:, i] < 0, z[:, i] * self.left_slopes[i], z[:,i])
        return z

    def prime(self, z):
        for i in range(self.n_neurons):
            z[:, i] = np.where(z[:, i] < 0,  self.left_slopes[i], 1)
        return z


class SparseAlternatedReLU:
    def __init__(self, slope):
        self.slope = slope

    def activation(self, z):
        z = np.where(z < 0, self.slope * z, z)
        return z

    def prime(self, z):
        z = np.where(z < 0, self.slope, 1)
        return z

class SparseAlternatedReLU2:
    def __init__(self, n):
        self.al = []
        for i in range(n):
            if i%2==0:
                self.al.append(0.75)
            else:
                self.al.append(-0.75)

    def activation(self, z):
        z = np.where(z < 0, self.al * z, z)
        return z

    def prime(self, z):
        z = np.where(z < 0, self.al, 1)
        return z



class RunningMeanReLU:
    def __init__(self):
        self.mean = 0
        self.n_batches = 0

    def activation(self, z, test = False):
        # if self.n_batches == 0:
        #     s = skew(z, axis=0)
        #     self.al = np.where(s < 0, -0.75, 0.75)
        # if self.n_batches == 0:
        #     self.mean = z.mean(axis=0)
        #     self.n_batches += 1
        # else:
        #     self.n_batches += 1
        #     self.mean = (self.mean + z.mean(axis=0)) / 2

        # s = skew(z, axis=0)
        if not test:
            p95 = np.percentile(z, 95, axis=0)
            p5 = np.percentile(z, 5, axis=0)
            diff = np.abs(np.abs(p5) - np.abs(p95))
            self.al = np.where(diff > 0, -0.5, 0.5)
            self.tl = - diff

        z = np.where(z > self.tl, z,  z * self.al)
        return z

    def prime(self, z):
        z = np.where(z > self.tl, 1, self.al)
        return z


class Swish:
    @staticmethod
    def activation(z):
        z = z * Sigmoid.activation(z)
        return z

    @staticmethod
    def prime(z):
        return Swish.activation(z) + Sigmoid.activation(z) * (1 - Swish.activation(z))


class Sigmoid:
    @staticmethod
    def activation(z):
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def prime(z):
        return Sigmoid.activation(z) * (1 - Sigmoid.activation(z))


class Softmax:
    @staticmethod
    def activation(z, test=False):
        """
        https://stackoverflow.com/questions/34968722/softmax-function-python
        Numerically stable version
        """
        exps = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1, keepdims=True)


class CrossEntropy:
    """
    Used with Softmax activation in final layer
    """

    def delta(self, y_true, y_pred):
        """
        Back propagation error delta
        :return: (array)
        """
        return y_pred - y_true

    @staticmethod
    def loss(y_true, y):
        """
        https://datascience.stackexchange.com/questions/9302/the-cross-entropy-error-function-in-neural-networks
        :param y_true: (array) One hot encoded truth vector.
        :param y: (array) Prediction vector
        :return: (flt)
        """
        y /= y.sum(axis=-1, keepdims=True)
        output = np.clip(y, 1e-7, 1 - 1e-7)
        return np.sum(y_true * - np.log(output), axis=-1).sum() / y.shape[0]


class CrossEntropyWeighted:
    """
    Used with Softmax activation in final layer
    """
    def __init__(self, class_weights):
        self.class_weights = class_weights

    def delta(self, y_true, y_pred):
        """
        Back propagation error delta
        :return: (array)
        """
        return y_pred - y_true

    def loss(self, y_true, y):
        """
        https://datascience.stackexchange.com/questions/9302/the-cross-entropy-error-function-in-neural-networks
        :param y_true: (array) One hot encoded truth vector.
        :param y: (array) Prediction vector
        :return: (flt)
        """
        return -np.mean(y_true * np.log(y + 1e-8) )

class RunningMeanReLU2:
    def __init__(self):
        self.mean = 0
        self.n_batches = 0
        self.n_batches_per_epoch = 50000/128

    def activation(self, z):
        if self.n_batches == 0:
            self.mean = z.mean(axis=0)
        else:
            self.mean = (self.mean + z.mean(axis=0)) / 2

        if self.n_batches / self.n_batches_per_epoch >= 100:
            p95 = np.percentile(z, 95, axis=0)
            p5 = np.percentile(z, 5, axis=0)
            self.al = np.where(np.abs(p5) - np.abs(p95) > 0, -0.75, 0.75)
            z = np.where(z > self.mean + 1e7, z, z * self.al)

        self.n_batches += 1
        return z

    def prime(self, z):
        if self.n_batches / self.n_batches_per_epoch >= 100:
            z = np.where(z > self.mean + 1e7, 1, self.al)
        else:
            z = np.where(z > 0, 1, 0)
        return z


class MSE:
    def __init__(self, activation_fn=None):
        """

        :param activation_fn: Class object of the activation function.
        """
        if activation_fn:
            self.activation_fn = activation_fn
        else:
            self.activation_fn = NoActivation

    def activation(self, z):
        return self.activation_fn.activation(z)

    @staticmethod
    def loss(y_true, y_pred):
        """
        :param y_true: (array) One hot encoded truth vector.
        :param y_pred: (array) Prediction vector
        :return: (flt)
        """
        return np.mean((y_pred - y_true) ** 2)

    @staticmethod
    def prime(y_true, y_pred):
        return y_pred - y_true

    def delta(self, y_true, y_pred):
        """
        Back propagation error delta
        :return: (array)
        """
        return self.prime(y_true, y_pred) * self.activation_fn.prime(y_pred)


class CrossEntropyL1:
    """
    Used with Softmax activation in final layer
    """

    def delta(self, y_true, y_pred):
        """
        Back propagation error delta
        :return: (array)
        """
        return y_pred - y_true

    @staticmethod
    def loss(y_true, y):
        """
        https://datascience.stackexchange.com/questions/9302/the-cross-entropy-error-function-in-neural-networks
        :param y_true: (array) One hot encoded truth vector.
        :param y: (array) Prediction vector
        :return: (flt)
        """
        y /= y.sum(axis=-1, keepdims=True)
        output = np.clip(y, 1e-7, 1 - 1e-7)
        return np.sum(y_true * - np.log(output), axis=-1).sum() / y.shape[0]


class NoActivation:
    """
    This is a plugin function for no activation.

    f(x) = x * 1
    """

    @staticmethod
    def activation(z):
        """
        :param z: (array) w(x) + b
        :return: z (array)
        """
        return z

    @staticmethod
    def prime(z):
        """
        The prime of z * 1 = 1
        :param z: (array)
        :return: z': (array)
        """
        return np.ones_like(z)