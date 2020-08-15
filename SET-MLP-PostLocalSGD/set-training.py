import time
import argparse
from utils.load_data import *
from sklearn.utils import class_weight
from models.set_mlp_sequential import *


# **** change the warning level ****
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Training settings
parser = argparse.ArgumentParser(description='SET Parallel Training ')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=3000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=5, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--lr-rate-decay', type=float, default=0.0, metavar='LRD',
                    help='learning rate decay (default: 0)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--dropout-rate', type=float, default=0.3, metavar='D',
                    help='Dropout rate')
parser.add_argument('--weight-decay', type=float, default=0.000, metavar='W',
                    help='Dropout rate')
parser.add_argument('--epsilon', type=int, default=20, metavar='E',
                    help='Sparsity level')
parser.add_argument('--zeta', type=float, default=0.3, metavar='Z',
                    help='It gives the percentage of unimportant connections which are removed and replaced with '
                         'random ones after every epoch(in [0..1])')
parser.add_argument('--n-neurons', type=int, default=3000, metavar='H',
                    help='Number of neurons in the hidden layer')
parser.add_argument('--seed', type=int, default=0, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--n-training-samples', type=int, default=50000, metavar='N',
                    help='Number of training samples')
parser.add_argument('--n-testing-samples', type=int, default=10000, metavar='N',
                    help='Number of testing samples')
parser.add_argument('--n-validation-samples', type=int, default=10000, help='Number of validation samples')
parser.add_argument('--dataset', default='higgs', help='Specify dataset. One of "cifar10", "fashionmnist"),'
                                                         '"higgs", "svhn", "madalon", "leukemia", "cllsub111",'
                                                         '"gli85", "smkcan187", "eurostat", "orlraws10p", "eurosat" or "mnist"')

if __name__ == "__main__":
    args = parser.parse_args()

    for i in range(5):
        # Set parameters
        n_hidden_neurons = args.n_neurons
        epsilon = args.epsilon
        zeta = args.zeta
        n_epochs = args.epochs
        batch_size = args.batch_size
        dropout_rate = args.dropout_rate
        learning_rate = args.lr
        momentum = args.momentum
        weight_decay = args.weight_decay
        n_training_samples = args.n_training_samples
        n_testing_samples = args.n_testing_samples
        learning_rate_decay = args.lr_rate_decay

        np.random.seed(i)
        class_weights = None
        weight_init = 'xavier'

        # Load dataset
        start_time = time.time()
        # Model architecture
        if args.dataset == 'higgs':
            # Model architecture higgs
            dimensions = (28, 1000, 1000, 1000, 2)
            loss = 'cross_entropy'
            weight_init = 'xavier'
            activations = (Relu, Relu, Relu, Softmax)
            X_train, Y_train, X_test, Y_test = load_higgs_data()
        elif args.dataset == 'fashionmnist' or args.dataset == 'mnist':
            # Model architecture mnist
            dimensions = (784, 1000, 1000, 1000, 10)
            loss = 'cross_entropy'
            activations = (Relu, Relu, Relu, Softmax)
            if args.dataset == 'fashionmnist':
                X_train, Y_train, X_test, Y_test = load_fashion_mnist_data(n_training_samples, n_testing_samples)
            else:
                X_train, Y_train, X_test, Y_test = load_mnist_data(n_training_samples, n_testing_samples)
        elif args.dataset == 'madalon':
            # Model architecture madalon
            dimensions = (500, 400, 100, 400, 1)
            loss = 'mse'
            activations = (Relu, Relu, Relu, Sigmoid)
            X_train, Y_train, X_test, Y_test = load_madelon_data()
        elif args.dataset == 'svhn':
            # Model architecture svhn
            dimensions = (3072, 4000, 1000, 4000, 10)
            loss = 'cross_entropy'
            activations = (Relu, Relu, Relu, Softmax)
            X_train, Y_train, X_test, Y_test = load_svhn_data(n_training_samples, n_testing_samples)
        elif args.dataset == 'cllsub111':
            # Model architecture cllsub111
            dimensions = (11340, 10000, 10000, 10000, 3)
            loss = 'cross_entropy'
            activations = (Relu, Relu, Relu, Softmax)
            X_train, Y_train, X_test, Y_test = load_cll_sub_111_data()
        elif args.dataset == 'gli85':
            # Model architecture gli85
            dimensions = (22283, 20000, 5000, 20000, 1)
            loss = 'mse'
            activations = (Relu, Relu, Relu, Sigmoid)
            X_train, Y_train, X_test, Y_test = load_gli_85_data()
        elif args.dataset == 'smkcan187':
            # Model architecture smkcan187
            dimensions = (22283, 20000, 5000, 20000, 1)
            activations = (Relu, Relu, Relu, Sigmoid)
            loss = 'mse'
            X_train, Y_train, X_test, Y_test = load_smk_can_187_data()
        elif args.dataset == 'orlraws10p':
            # Model architecture orlraws10p
            dimensions = (10304, 10000, 10000, 10000, 10)
            loss = 'cross_entropy'
            activations = (Relu, Relu, Relu, Softmax)
            X_train, Y_train, X_test, Y_test = load_orlraws_10P_data(n_training_samples, n_testing_samples)
        elif args.dataset == 'leukemia':
            batch_size = 5
            learning_rate = 0.005
            # Model architecture leukemia
            dimensions = (54675, 10000, 18)
            loss = 'cross_entropy'
            epsilon = 10
            activations = (Relu, Softmax)
            X_train, Y_train, X_test, Y_test = load_leukemia_data()
            class_weights = class_weight.compute_class_weight('balanced', np.unique(Y_train), Y_train)
            Y_train = np_utils.to_categorical(Y_train, 18)
            Y_test = np_utils.to_categorical(Y_test, 18)
        elif args.dataset == 'eurosat':
            dimensions = (64*64*3, 10000, 5000, 10000, 10)
            loss = 'cross_entropy'
            epsilon = 10
            batch_size = 64
            learning_rate = 0.005
            activations = (Relu, Relu, Relu, Softmax)
            X_train, Y_train, X_test, Y_test = load_eurosat__data()
        else:
            # Model architecture cifar10
            dimensions = (3072, 4000, 1000, 4000, 10)
            loss = 'cross_entropy'
            activations = (Relu, Relu, Relu, Softmax)
            X_train, Y_train, X_test, Y_test = load_cifar10_data_not_flattened(n_training_samples, n_testing_samples)

            # Prepare config object for the parameter server
        config = {
                'n_epochs': n_epochs,
                'batch_size': batch_size,
                'dropout_rate': dropout_rate,
                'seed': i,
                'lr': learning_rate,
                'lr_decay': learning_rate_decay,
                'zeta': zeta,
                'epsilon': epsilon,
                'momentum': momentum,
                'weight_decay': weight_decay,
                'n_hidden_neurons': n_hidden_neurons,
                'n_training_samples': n_training_samples,
                'n_testing_samples': n_testing_samples,
                'loss': loss,
                'weight_init': weight_init
        }

        step_time = time.time() - start_time
        print("Loading augmented dataset time: ", step_time)

        # Load basic cifar10 dataset
        # X_train, Y_train, X_test, Y_test = load_cifar10_data(n_training_samples, n_testing_samples)

        # Create SET-MLP (MLP with adaptive sparse connectivity trained with Sparse Evolutionary Training)
        # print("Number of neurons per layer:", dimensions[0], dimensions[1], dimensions[2],
        #       dimensions[3], dimensions[4])

        set_mlp = SET_MLP(dimensions, activations, class_weights=class_weights, **config)
        start_time = time.time()
        set_mlp.fit(X_train, Y_train, X_test, Y_test, testing=True,
                    save_filename=r"Experiments/relu_set_mlp_sequential_" + args.dataset + "_" +
                                  str(n_training_samples) + "_training_samples_e" + str(
                        epsilon) + "_rand" + str(i) + "_epochs" + str(n_epochs) + "_batchsize" + str(batch_size)
                    )
        step_time = time.time() - start_time
        print("\nTotal training time: ", step_time)
        print("\nTraining time: ", set_mlp.training_time)
        print("\nTesting time: ", set_mlp.testing_time)
        print("\nEvolution time: ", set_mlp.evolution_time)

        # Test SET-MLP
        if args.dataset == 'cifar10':
            accuracy, _ = set_mlp.predict(X_test.reshape(-1, 32 * 32 * 3), Y_test, batch_size=1)
        else:
            accuracy, _ = set_mlp.predict(X_test, Y_test, batch_size=1)
        print("\nAccuracy of the last epoch on the testing data: ", accuracy)
