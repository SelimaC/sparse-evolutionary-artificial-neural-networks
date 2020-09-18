import argparse
import logging
import pandas as pd
from utils.nn_functions import *
from mpi4py import MPI
from time import time
from utils.load_data import *
from mpi_training.mpi.manager import MPIManager
from mpi_training.train.algo import Algo
from mpi_training.train.data import Data
from mpi_training.train.model import SETMPIModel
from mpi_training.logger import initialize_logger
from sklearn.utils import class_weight

# Run this file with "mpiexec -n 6 python MPI_training.py --synchronous"
# Add --synchronous if you want to train in synchronous mode
# Add --monitor to enable cpu and memory monitoring

# Uncomment next lines for debugging with size > 1 (note that port mapping ids change at very run)
# size = MPI.COMM_WORLD.Get_size()
# rank = MPI.COMM_WORLD.Get_rank()
# import pydevd_pycharm
# port_mapping = [56131, 56135] # Add ids of processes you want to debug in this list
# pydevd_pycharm.settrace('localhost', port=port_mapping[rank], stdoutToServer=True, stderrToServer=True)


def shared_partitions(n, num_workers, batch_size):
    """"
    Split the training dataset equally amongst the workers
    """
    dinds = list(range(n))
    num_batches = n // batch_size
    worker_size = num_batches // num_workers

    data = dict.fromkeys(list(range(num_workers)))

    for w in range(num_workers):
        data[w] = dinds[w * batch_size * worker_size: (w+1) * batch_size * worker_size]

    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--monitor', help='Monitor cpu and gpu utilization', default=False, action='store_true')

    # Configuration of network topology
    parser.add_argument('--masters', help='number of master processes', default=1, type=int)
    parser.add_argument('--processes', help='number of processes per worker', default=1, type=int)
    parser.add_argument('--synchronous', help='run in synchronous mode', action='store_true')

    # Configuration of training process
    parser.add_argument('--loss', help='loss function', default='cross_entropy')
    parser.add_argument('--sync-every', help='how often to sync weights with master',
                        default=1, type=int, dest='sync_every')
    parser.add_argument('--mode', help='Mode of operation.'
                        'One of "sgd" (Stohastic Gradient Descent), "sgdm" (Stohastic Gradient Descent with Momentum),'
                        '"easgd" (Elastic Averaging SGD) or "gem" (Gradient Energy Matching)',
                        default='sgdm')
    parser.add_argument('--elastic-force', help='beta parameter for EASGD', type=float, default=0.9)
    parser.add_argument('--elastic-lr', help='worker SGD learning rate for EASGD',
                        type=float, default=1.0, dest='elastic_lr')
    parser.add_argument('--elastic-momentum', help='worker SGD momentum for EASGD',
                        type=float, default=0, dest='elastic_momentum')
    parser.add_argument('--gem-lr', help='learning rate for GEM', type=float, default=0.01, dest='gem_lr')
    parser.add_argument('--gem-momentum', help='momentum for GEM', type=float, default=0.9, dest='gem_momentum')
    parser.add_argument('--gem-kappa', help='Proxy amplification parameter for GEM', type=float, default=2.0,
                        dest='gem_kappa')

    # logging configuration
    parser.add_argument('--log-file', default=None, dest='log_file',
                        help='log file to write, in additon to output stream')
    parser.add_argument('--log-level', default='info', dest='log_level', help='log level (debug, info, warn, error)')

    # Model configuration
    parser.add_argument('--batch-size', type=int, default=128, help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=500,  help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate (default: 0.01)')
    parser.add_argument('--lr-rate-decay', type=float, default=0.0, help='learning rate decay (default: 0)')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum (default: 0.5)')
    parser.add_argument('--dropout-rate', type=float, default=0.3, help='Dropout rate')
    parser.add_argument('--weight-decay', type=float, default=0.0, help='Weight decay (l2 regularization)')
    parser.add_argument('--epsilon', type=int, default=20, help='Sparsity level')
    parser.add_argument('--zeta', type=float, default=0.3,
                        help='It gives the percentage of unimportant connections which are removed and replaced with '
                             'random ones after every epoch(in [0..1])')
    parser.add_argument('--n-neurons', type=int, default=3000, help='Number of neurons in the hidden layer')
    parser.add_argument('--seed', type=int, default=4, help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10,
                        help='how many batches to wait before logging training status')
    parser.add_argument('--n-training-samples', type=int, default=50000, help='Number of training samples')
    parser.add_argument('--n-testing-samples', type=int, default=10000, help='Number of testing samples')
    parser.add_argument('--augmentation', default=True, help='Data augmentation', action='store_true')
    parser.add_argument('--dataset', default='cifar10', help='Specify dataset. One of "cifar10", "fashionmnist"),'
                                                             '"higgs", "svhn", "madelon", "leukemia", "cllsub111",'
                                                             '"gli85", "smkcan187", "eurostat", "orlraws10p" or "mnist"')

    args = parser.parse_args()

    weight_init = 'xavier'
    prune = False
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
    class_weights = None

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
        batch_size = 128
        learning_rate = 0.01
        epsilon = 20
        weight_init = 'he_uniform'
        #activations = (Relu, Relu, Relu, Softmax)
        activations = (SparseAlternatedReLU(-0.5), SparseAlternatedReLU(0.5), SparseAlternatedReLU(-0.5), Softmax)
        if args.dataset == 'fashionmnist':
            X_train, Y_train, X_test, Y_test = load_fashion_mnist_data(60000, 10000)
        else:
            X_train, Y_train, X_test, Y_test = load_mnist_data(args.n_training_samples, args.n_testing_samples)
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
        X_train, Y_train, X_test, Y_test = load_svhn_data(args.n_training_samples, args.n_testing_samples)
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
        X_train, Y_train, X_test, Y_test = load_orlraws_10P_data()
    elif args.dataset == 'leukemia':
        batch_size = 5
        learning_rate = 0.005
        # Model architecture leukemia
        dimensions = (54675, 27500, 5000, 27500, 18)

        loss = 'cross_entropy_weighted'
        dropout_rate = 0.3
        weight_init = 'normal'
        epsilon = 10
        activations = (Relu, Relu, Relu, Softmax)
        X_train, Y_train, X_test, Y_test = load_leukemia_data()
        class_weights = class_weight.compute_class_weight('balanced', np.unique(Y_train), Y_train)
        Y_train = np_utils.to_categorical(Y_train, 18)
        Y_test = np_utils.to_categorical(Y_test, 18)

    elif args.dataset == 'eurosat':
        dimensions = (64 * 64 * 3, 10000, 5000, 10000, 10)
        loss = 'cross_entropy'
        epsilon = 10
        batch_size = 64
        learning_rate = 0.005
        activations = (Relu, Relu, Relu, Softmax)
        X_train, Y_train, X_test, Y_test = load_eurosat__data()
    else:
        # Model architecture cifar10
        dimensions = (3072, 4000, 1000, 4000, 10)
        weight_init = 'he_uniform'
        loss = 'cross_entropy'
        learning_rate = 0.01
        batch_size=128
        epsilon=20
        activations = (SparseAlternatedReLU(-0.75), SparseAlternatedReLU(0.75), SparseAlternatedReLU(-0.75), Softmax)
        if args.augmentation:
            X_train, Y_train, X_test, Y_test = load_cifar10_data_not_flattened(50000, 10000)
        else:
            X_train, Y_train, X_test, Y_test = load_cifar10_data(50000, 10000)


    # SET parameters
    model_config = {
        'n_epochs': n_epochs,
        'batch_size': batch_size,
        'dropout_rate': dropout_rate,
        'lr': learning_rate,
        'zeta': zeta,
        'epsilon': epsilon,
        'momentum': momentum,
        'weight_decay': 0.0,
        'n_hidden_neurons': n_hidden_neurons,
        'n_training_samples': n_training_samples,
        'n_testing_samples': n_testing_samples,
        'loss': loss,
        'weight_init': weight_init,
        'prune': prune
    }

    # Comment this if you would like to use the full power of randomization. I use it to have repeatable results.
    np.random.seed(args.seed)

    comm = MPI.COMM_WORLD.Dup()

    model_weights = None
    rank = comm.Get_rank()
    num_processes = comm.Get_size()
    num_workers = num_processes - 1

    # Initialize logger
    base_file_name = "ParallelResults/alrelu75_augmented_sgdm_no_pruning_async_batch128_set_mpi_"+ str(args.dataset)+"_" + str(args.epochs) + "_epochs_e" + \
                    str(args.epsilon) + "_rand" + str(args.seed) + "_num_workers_" + str(num_workers)
    log_file = base_file_name + "_logs_execution.txt"

    save_filename = base_file_name + "_process_" + str(rank)

    initialize_logger(filename=log_file, file_level=args.log_level, stream_level=args.log_level)

    if num_processes == 1:
        validate_every = int(X_train.shape[0] // (args.batch_size * args.sync_every))
        data = Data(batch_size=args.batch_size,
                    x_train=X_train, y_train=Y_train,
                    x_test=X_test, y_test=Y_test, augmentation=True,
                    dataset=args.dataset)
    else:
        if rank != 0:
            validate_every = int(X_train.shape[0] // (args.batch_size * args.sync_every))
            partitions = shared_partitions(X_train.shape[0], num_workers, args.batch_size)
            data = Data(batch_size=args.batch_size,
                        x_train=X_train[partitions[rank - 1]], y_train=Y_train[partitions[rank - 1]],
                        x_test=X_test, y_test=Y_test, augmentation=args.augmentation,
                        dataset=args.dataset)
            logging.info(f"Data partition contains {data.x_train.shape[0]} samples")
        else:
            validate_every = int(X_train.shape[0] // args.batch_size)
            if args.synchronous:
                validate_every = int(X_train.shape[0] // (args.batch_size * num_workers))
            data = Data(batch_size=args.batch_size,
                        x_train=X_train, y_train=Y_train,
                        x_test=X_test, y_test=Y_test, augmentation=args.augmentation,
                        dataset=args.dataset)
            logging.info(f"Validate every {validate_every} time steps")
    del X_train, Y_train, X_test, Y_test

    # Scale up the learning rate for synchronous training
    if args.synchronous:
        args.lr = args.lr * (num_workers * 0.75)

    # Some input arguments may be ignored depending on chosen algorithm
    if args.mode == 'easgd':
        algo = Algo(None, loss=args.loss, validate_every=validate_every,
                    mode='easgd', sync_every=args.sync_every,
                    elastic_force=args.elastic_force / num_workers,
                    elastic_lr=args.elastic_lr, lr=args.lr,
                    elastic_momentum=args.elastic_momentum)
    elif args.mode == 'gem':
        algo = Algo('gem', validate_every=validate_every,
                    mode='gem', sync_every=args.sync_every,
                    learning_rate=args.gem_lr, momentum=args.gem_momentum, kappa=args.gem_kappa)
    elif args.mode == 'sgdm':
        algo = Algo(optimizer='sgdm', validate_every=validate_every, lr=args.lr,
                    sync_every=args.sync_every, weight_decay=args.weight_decay, momentum=args.gem_momentum, n_workers=num_workers)
    else:
        algo = Algo(optimizer='sgd', validate_every=validate_every, lr=args.lr, sync_every=args.sync_every)

    # Instantiate SET model
    # Instantiate SET model
    if rank == 0:
        from models.set_mlp_mpi_master import *

        model = SETMPIModel(dimensions, activations, class_weights=class_weights,**model_config)
    else:
        from models.set_mlp_mpi import *

        model = SETMPIModel(dimensions, activations, class_weights=class_weights, **model_config)

    # Creating the MPIManager object causes all needed worker and master nodes to be created
    manager = MPIManager(comm=comm, data=data, algo=algo, model=model,
                         num_epochs=args.epochs, num_masters=args.masters,
                         num_processes=args.processes, synchronous=args.synchronous,
                         monitor=args.monitor, save_filename=save_filename)

    # Process 0 launches the training procedure
    if rank == 0:
        logging.debug('Training configuration: %s', algo.get_config())

        t_0 = time()
        histories = manager.process.train()
        delta_t = time() - t_0
        manager.free_comms()
        logging.info("Testing time is {0:.3f} seconds".format(manager.process.validate_time))
        delta_t -= manager.process.validate_time
        logging.info("Training finished in {0:.3f} seconds".format(delta_t))
        logging.info("Evolution time is {0:.3f} seconds".format(manager.process.evolution_time))

        logging.info("------------------------------------------------------------------------------------------------")
        logging.info("Final performance of the model on the test dataset")
        manager.process.validate()

    comm.barrier()
    logging.info("Terminating")
