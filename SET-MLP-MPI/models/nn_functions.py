import numpy as np


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

class ReluShifted:
    @staticmethod
    def activation(z):
        z[z < -0.05] = 0
        return z

    @staticmethod
    def prime(z):
        z[z < -0.05] = 0
        z[z > -0.05] = 1
        return z


class ReluActive:
    @staticmethod
    def activation(z, probs):
        z[z < 0] = 0

        for i in range(len(probs)):
           z[:,i] *= probs[i]
        return z

    @staticmethod
    def prime(z, probs):
        z[z < 0] = 0
        for i in range(len(probs)):
            z[:, i] = probs[i]
        return z


class RReLu:
    def __init__(self, n_neurons=None):
        self.n_neurons = n_neurons
        self.left_slopes = np.random.uniform(-1, 1, self.n_neurons)

    def activation(self, z):
        for i in range(self.n_neurons):
            x = z[:,i]
            x[x < 0] *= self.left_slopes[i]
        return z

    def prime(self, z):
        for i in range(self.n_neurons):
            x = z[:,i]
            x[x < 0] = self.left_slopes[i]
        #z[z < 0] = self.left_slopes
        z[z > 0] = 1
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


class MeanReLu:
    @staticmethod
    def activation(z):
        z[z < z.mean()] = 0
        return z

    @staticmethod
    def prime(z):
        z_mean = z.mean()
        z[z < z_mean] = 0
        z[z > z.mean()] = 1
        return z


class SparseRelu:
    @staticmethod
    def activation(z):
        z = np.where(z < 0, -1/4 * z, z)
        return z

    @staticmethod
    def prime(z):
        z = np.where(z < 0, -1/4, 1)
        return z


class SparseReluSym:
    @staticmethod
    def activation(z):
        z = np.where(z < 0, - z, z)
        return z

    @staticmethod
    def prime(z):
        z = np.where(z < 0, -1, 1)
        return z


class SparseReluHalfSym:
    @staticmethod
    def activation(z):
        z = np.where(z < 0, - 0.5 * z, z)
        return z

    @staticmethod
    def prime(z):
        z = np.where(z < 0, -0.5, 1)
        return z


class SparseReluHalfSymPositive:
    @staticmethod
    def activation(z):
        z = np.where(z < 0, 0.5 * z, z)
        return z

    @staticmethod
    def prime(z):
        z = np.where(z < 0, 0.5, 1)
        return z


class SparseReluSym75:
    @staticmethod
    def activation(z):
        z = np.where(z < 0, - 0.75 * z, z)
        return z

    @staticmethod
    def prime(z):
        z = np.where(z < 0, -0.75, 1)
        return z


class SparseReluSym75Positive:
    @staticmethod
    def activation(z):
        z = np.where(z < 0, 0.75 * z, z)
        return z

    @staticmethod
    def prime(z):
        z = np.where(z < 0, 0.75, 1)
        return z


class SparseReluSym25Positive:
    @staticmethod
    def activation(z):
        z = np.where(z < 0, 0.25 * z, z)
        return z

    @staticmethod
    def prime(z):
        z = np.where(z < 0, 0.25, 1)
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
    def activation(z):
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