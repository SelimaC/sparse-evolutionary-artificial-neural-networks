from set_mlp import *
from parameter_server import *
import time
import argparse

# Training settings
parser = argparse.ArgumentParser(description='SET Parallel Training ')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=3000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.05, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--dropout-rate', type=float, default=0.2, metavar='D',
                    help='Dropout rate')
parser.add_argument('--weight-decay', type=float, default=0.0002, metavar='W',
                    help='Dropout rate')
parser.add_argument('--epsilon', type=int, default=13, metavar='E',
                    help='Sparsity level')
parser.add_argument('--zeta', type=float, default=0.3, metavar='Z',
                    help='It gives the percentage of unimportant connections which are removed and replaced with '
                         'random ones after every epoch(in [0..1])')
parser.add_argument('--n-neurons', type=int, default=3000, metavar='H',
                    help='Number of neurons in the hidden layer')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--n-processes', type=int, default=1, metavar='N',
                    help='how many training processes to use (default: 2)')
parser.add_argument('--cuda', action='store_true', default=False,
                    help='enables CUDA training')

if __name__ == "__main__":
    args = parser.parse_args()


    for i in range(1):

        #load data
        noTrainingSamples = 5000 # max 60000 for Fashion MNIST
        noTestingSamples = 1000  # max 10000 for Fshion MNIST
        X_train, Y_train, X_test, Y_test = load_cifar10_data(noTrainingSamples, noTestingSamples)

        #set model parameters
        n_hidden_neurons = args.n_neurons
        epsilon = args.epsilon
        zeta = args.zeta
        n_epochs = args.epochs
        batch_size = args.batch_size
        dropout_rate = args.dropout_rate
        learning_rate = args.lr
        momentum = args.momentum
        weight_decay = args.weight_decay
        n_processes = args.n_processes

        config = {
            'n_processes': n_processes,
            'n_epochs': n_epochs,
            'delay': 1,
            'delaytype': 'const',  # const or random
            'batch_size': batch_size,
            'dropout_rate': dropout_rate,
            'seed': i,
            'lr': learning_rate,
            'zeta': zeta,
            'epsilon': epsilon,
            'momentum': momentum,
            'weight_decay': weight_decay,
            'n_hidden_neurons': n_hidden_neurons
        }

        np.random.seed(i)

        # create SET-MLP (MLP with adaptive sparse connectivity trained with Sparse Evolutionary Training)
        print("Number of neurons per layer:", X_train.shape[1], n_hidden_neurons, n_hidden_neurons, n_hidden_neurons, Y_train.shape[1])


        # set_mlp = SET_MLP((X_train.shape[1], noHiddenNeuronsLayer, noHiddenNeuronsLayer, noHiddenNeuronsLayer,
        #                    Y_train.shape[1]), (Relu, Relu, Relu, Sigmoid), MSE, epsilon=epsilon)
        #
        # # train SET-MLP
        # set_mlp.fit(X_train, Y_train, X_test, Y_test, loss=MSE, epochs=noTrainingEpochs, batch_size=batchSize,
        #             learning_rate=learningRate,
        #             momentum=momentum, weight_decay=weightDecay, zeta=zeta, dropoutrate=dropoutRate, testing=True,
        #             save_filename="Results/set_mlp_" + str(noTrainingSamples) + "_training_samples_e" + str(
        #                 epsilon) + "_rand" + str(i))

        ps = ParameterServer(**config)
        ps.train()

        # test SET-MLP

        # accuracy, _ = ps.predict(X_test, Y_test, batch_size=1)

        # print("\nAccuracy of the last epoch on the testing data: ", accuracy)
