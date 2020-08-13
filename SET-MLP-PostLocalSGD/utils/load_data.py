### Utilities for mpi_learn module
import os
import numpy as np
import pandas as pd
import os
#import GEOparse
from concurrent.futures import ProcessPoolExecutor
from keras.datasets import cifar10, mnist
from keras.utils import np_utils
from PIL import Image
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Augmented dataset path
cur_dir = os.path.dirname(os.path.abspath(__file__))
path_to_data = ['..', '..', 'data', 'EuroSAT']
images_dirs = os.path.join(cur_dir, *path_to_data)


class Error(Exception):
    pass


# The Leukemia dataset is obtained from the NCBI GEO repository with the accession number GSE13159
def load_leukemia_data(n_training_samples=1397, n_testing_samples=699):
    labels = np.loadtxt("../data/Leukemia/labels.txt")
    data = np.loadtxt("../data/Leukemia/values.txt")

    index_data = np.arange(data.shape[0])
    np.random.shuffle(index_data)

    data = data[index_data, :]
    labels = labels[index_data]

    label_0 = np.argwhere(labels == 0)
    label_0_train_ids = label_0[0:26]; label_0_test_ids = label_0[26:]

    label_1 = np.argwhere(labels == 1)
    label_1_train_ids = label_1[0:39]; label_1_test_ids = label_1[39:]

    label_2 = np.argwhere(labels == 2)
    label_2_train_ids = label_2[0:24]; label_2_test_ids = label_2[24:]

    label_3 = np.argwhere(labels == 3)
    label_3_train_ids = label_3[0:30]; label_3_test_ids = label_3[30:]

    label_4 = np.argwhere(labels == 4)
    label_4_train_ids = label_4[0:19]; label_4_test_ids = label_4[19:]

    label_5 = np.argwhere(labels == 5)
    label_5_train_ids = label_5[0:236]; label_5_test_ids = label_5[236:]

    label_6 = np.argwhere(labels == 6)
    label_6_train_ids = label_6[0:25]; label_6_test_ids = label_6[25:]

    label_7 = np.argwhere(labels == 7)
    label_7_train_ids = label_7[0:25]; label_7_test_ids = label_7[25:]

    label_8 = np.argwhere(labels == 8)
    label_8_train_ids = label_8[0:26]; label_8_test_ids = label_8[26:]

    label_9 = np.argwhere(labels == 9)
    label_9_train_ids = label_9[0:299]; label_9_test_ids = label_9[299:]

    label_10 = np.argwhere(labels == 10)
    label_10_train_ids = label_10[0:51]; label_10_test_ids = label_10[51:]

    label_11 = np.argwhere(labels == 11)
    label_11_train_ids = label_11[0:138]; label_11_test_ids = label_11[138:]

    label_12 = np.argwhere(labels == 12)
    label_12_train_ids = label_12[0:48]; label_12_test_ids = label_12[48:]

    label_13 = np.argwhere(labels == 13)
    label_13_train_ids = label_13[0:47]; label_13_test_ids = label_13[47:]

    label_14 = np.argwhere(labels == 14)
    label_14_train_ids = label_14[0:116]; label_14_test_ids = label_14[116:]

    label_15 = np.argwhere(labels == 15)
    label_15_train_ids = label_15[0:81]; label_15_test_ids = label_15[81:]

    label_16 = np.argwhere(labels == 16)
    label_16_train_ids = label_16[0:158]; label_16_test_ids = label_16[158:]

    label_17 = np.argwhere(labels == 17)
    label_17_train_ids = label_17[0:9]; label_17_test_ids = label_17[9:]

    # replace nan with col means
    data = np.where(np.isnan(data), np.ma.array(data, mask=np.isnan(data)).mean(axis=0), data)

    labels = np.array(labels)
    data = np.array(data)

    x_test = np.vstack((data[label_0_test_ids.reshape(-1, )],
                         data[label_1_test_ids.reshape(-1,)],
                         data[label_2_test_ids.reshape(-1,)],
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
                         data[label_17_test_ids.reshape(-1, )]))
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
                        labels[label_17_test_ids.reshape(-1, )]))
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
                        data[label_17_train_ids.reshape(-1, )]))
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
                         labels[label_17_train_ids.reshape(-1, )]))

    mn, mx = x_train.min(), x_train.max()
    x_train = (x_train - mn) / (mx - mn)
    x_test = (x_test - mn) / (mx - mn)
    #
    # xTrainMean = np.mean(x_train, axis=0)
    # xTtrainStd = np.std(x_train, axis=0)
    # x_train = (x_train - xTrainMean) / xTtrainStd
    # x_test = (x_test - xTrainMean) / xTtrainStd
    # x_test = (x_test - xTrainMean) / xTtrainStd

    # x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.33, random_state=42)
    # y_train = np_utils.to_categorical(y_train, 18)
    # y_test = np_utils.to_categorical(y_test, 18)

    return x_train, y_train, x_test, y_test


def extract_leukemia_data():
    values_ds = []
    values = []
    classes = []
    gse_obj = GEOparse.get_GEO(filepath="../data/Leukemia/GSE13159_family.soft.gz")
    for key, gsms in gse_obj.gsms.items():
        metadata = gsms.metadata
        class_leukemia = metadata['characteristics_ch1'][1][16:]
        value_ds = np.array(gsms.table.VALUE_DS)
        value = np.array(gsms.table.VALUE)
        classes.append(class_leukemia)
        values.append(value)
        values_ds.append(value_ds)
    values = np.asarray(values)
    values_ds = np.asarray(values_ds)


    np.savetxt("../data/Leukemia/values.txt", values)
    np.savetxt("../data/Leukemia/values_ds.txt", values_ds)

    print(np.unique(classes))
    # creating instance of labelencoder
    labelencoder = LabelEncoder()
    classes = np.asarray(classes)
    classes = labelencoder.fit_transform(classes)
    np.savetxt("../data/Leukemia/labels.txt", classes)


    # Face recognition dataset


def load_orlraws_10P_data(n_training_samples=70, n_testing_samples=30):
    # Load the data

    data_raw = loadmat('../data/orlraws10P/orlraws10P.mat')

    # Load images and labels
    data = np.array(data_raw['X'])
    labels = data_raw['Y']

    # Convert data into 'float32' type
    data = data.astype('float32')
    labels = np_utils.to_categorical(labels-1, 10)

    index_data = np.arange(data.shape[0])
    np.random.shuffle(index_data)

    data = data[index_data, :]
    labels = labels[index_data, :]

    y_train = np.array(labels[0:70, :])
    x_train = np.array(data[0:70, :])

    y_test = np.array(labels[70:100, :])
    x_test = np.array(data[70:100, :])

    xTrainMean = np.mean(x_train, axis=0)
    xTtrainStd = np.std(x_train, axis=0)
    x_train = (x_train - xTrainMean) / xTtrainStd
    x_test = (x_test - xTrainMean) / xTtrainStd
    x_test = (x_test - xTrainMean) / xTtrainStd

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    return x_train, y_train, x_test, y_test


# Microarray, Bio
def load_smk_can_187_data(n_training_samples=124, n_testing_samples=63):
    # Load the data

    data_raw = loadmat('../data/SMK-CAN-187/SMK-CAN-187.mat')

    # Load images and labels
    data = np.array(data_raw['X'])
    labels = data_raw['Y']

    # Convert data into 'float32' type
    data = data.astype('float32')
    labels = (labels-1).reshape(187,)

    index_data = np.arange(data.shape[0])
    np.random.shuffle(index_data)

    data = data[index_data, :]
    labels = labels[index_data]

    y_train = np.array(labels[0:124])
    x_train = np.array(data[0:124, :])

    y_test = np.array(labels[124:187])
    x_test = np.array(data[124:187, :])

    xTrainMean = np.mean(x_train, axis=0)
    xTtrainStd = np.std(x_train, axis=0)
    x_train = (x_train - xTrainMean) / xTtrainStd
    x_test = (x_test - xTrainMean) / xTtrainStd
    x_test = (x_test - xTrainMean) / xTtrainStd

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    return x_train, y_train, x_test, y_test


# Microarray, Bio
def load_gli_85_data(n_training_samples=56, n_testing_samples=29):
    # Load the data

    data_raw = loadmat('../data/GLI-85/GLI-85.mat')

    # Load images and labels
    data = np.array(data_raw['X'])
    labels = data_raw['Y']

    # Convert data into 'float32' type
    data = data.astype('float32')
    labels = (labels-1).reshape(85,)

    index_data = np.arange(data.shape[0])
    np.random.shuffle(index_data)

    data = data[index_data, :]
    labels = labels[index_data]

    y_train = np.array(labels[0:56])
    x_train = np.array(data[0:56, :])

    y_test = np.array(labels[56:85])
    x_test = np.array(data[56:85, :])

    xTrainMean = np.mean(x_train, axis=0)
    xTtrainStd = np.std(x_train, axis=0)
    x_train = (x_train - xTrainMean) / xTtrainStd
    x_test = (x_test - xTrainMean) / xTtrainStd
    x_test = (x_test - xTrainMean) / xTtrainStd

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    return x_train, y_train, x_test, y_test


# Microarray, Bio
def load_cll_sub_111_data(n_training_samples=74, n_testing_samples=37):
    # Load the data

    data_raw = loadmat('../data/CLL-SUB-111/CLL-SUB-111.mat')

    # Load images and labels
    data = np.array(data_raw['X'])
    labels = data_raw['Y']

    # Convert data into 'float32' type
    data = data.astype('float32')
    labels = np_utils.to_categorical(labels-1, 3)

    index_data = np.arange(data.shape[0])
    np.random.shuffle(index_data)

    data = data[index_data, :]
    labels = labels[index_data, :]

    y_train = np.array(labels[0:74, :])
    x_train = np.array(data[0:74, :])

    y_test = np.array(labels[74:111, :])
    x_test = np.array(data[74:111, :])

    xTrainMean = np.mean(x_train, axis=0)
    xTtrainStd = np.std(x_train, axis=0)
    x_train = (x_train - xTrainMean) / xTtrainStd
    x_test = (x_test - xTrainMean) / xTtrainStd
    x_test = (x_test - xTrainMean) / xTtrainStd

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    return x_train, y_train, x_test, y_test


# Artificial dataset with two classes
def load_madelon_data():
    # Download the data
    x_train = np.loadtxt("../data/Madelon/madelon_train.data")
    y_train = np.loadtxt('../data/Madelon//madelon_train.labels')
    x_val = np.loadtxt('../data/Madelon/madelon_valid.data')
    y_val = np.loadtxt('../data/Madelon/madelon_valid.labels')
    x_test = np.loadtxt('../data/Madelon/madelon_test.data')

    y_train = np.where(y_train == -1, 0, 1)
    y_val = np.where(y_val == -1, 0, 1)

    xTrainMean = np.mean(x_train, axis=0)
    xTtrainStd = np.std(x_train, axis=0)
    x_train = (x_train - xTrainMean) / xTtrainStd
    x_test = (x_test - xTrainMean) / xTtrainStd
    x_val = (x_val - xTrainMean) / xTtrainStd

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_val = x_val.astype('float32')

    return x_train, y_train, x_val, y_val


# This is a classification problem to distinguish between a signal process which produces Higgs bosons and a background process which does not.
def load_higgs_data(n_training_samples=1050000, n_testing_samples=500000):
    N = 1050000.  # Change this line adjust the number of rows.
    data = pd.read_csv("../data/HIGGS/HIGGS.csv", nrows=N, header=None)
    test_data = pd.read_csv("../data/HIGGS/HIGGS.csv", nrows=500000, header=None, skiprows=1050000)

    y_train = np.array(data.loc[:, 0])
    x_train = np.array(data.loc[:, 1:])
    x_test = np.array(test_data.loc[:, 1:])
    y_test = np.array(test_data.loc[:, 0])

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    index_train = np.arange(x_train.shape[0])
    np.random.shuffle(index_train)

    index_test = np.arange(x_test.shape[0])
    np.random.shuffle(index_test)

    x_train = x_train[index_train[0:n_training_samples], :]
    y_train = y_train[index_train[0:n_training_samples]]

    x_test = x_test[index_test[0:n_testing_samples], :]
    y_test = y_test[index_test[0:n_testing_samples]]

    y_train = np_utils.to_categorical(y_train, 2)
    y_test = np_utils.to_categorical(y_test, 2)

    return x_train, y_train, x_test, y_test


# The Street View House Numbers (SVHN) Dataset
def load_svhn_data(n_training_samples, n_testing_samples):
    # Load the data

    train_raw = loadmat('../data/SVHN/train_32x32.mat')
    test_raw = loadmat('../data/SVHN/test_32x32.mat')

    # Load images and labels
    train_images = np.array(train_raw['X'])
    test_images = np.array(test_raw['X'])

    train_labels = train_raw['y']
    test_labels = test_raw['y']

    # Fix the axes of the images
    train_images = np.moveaxis(train_images, -1, 0)
    test_images = np.moveaxis(test_images, -1, 0)

    # Convert train and test images into 'float32' type

    train_images = train_images.astype('float32')
    test_images = test_images.astype('float32')

    # Convert train and test labels into categorical labels
    train_labels = np_utils.to_categorical(train_labels-1, 10)
    test_labels = np_utils.to_categorical(test_labels-1, 10)

    train_images /= 255.0
    test_images /= 255.0

    index_train = np.arange(train_images.shape[0])
    np.random.shuffle(index_train)

    index_test = np.arange(test_images.shape[0])
    np.random.shuffle(index_test)

    x_train = train_images[index_train[0:n_training_samples], :]
    y_train = train_labels[index_train[0:n_training_samples], :]

    x_test = test_images[index_test[0:n_testing_samples], :]
    y_test = test_labels[index_test[0:n_testing_samples], :]

    return x_train, y_train, x_test, y_test


# The MNIST database of handwritten digits.
def load_mnist_data(n_training_samples, n_testing_samples):

    # read CIFAR10 data
    (x, y), (x_test, y_test) = mnist.load_data()

    y = np_utils.to_categorical(y, 10)
    y_test = np_utils.to_categorical(y_test, 10)
    x = x.astype('float32')
    x_test = x_test.astype('float32')

    index_train = np.arange(x.shape[0])
    np.random.shuffle(index_train)

    index_test = np.arange(x_test.shape[0])
    np.random.shuffle(index_test)

    x_train = x[index_train[0:n_training_samples], :]
    y_train = y[index_train[0:n_training_samples], :]

    x_test = x_test[index_test[0:n_testing_samples], :]
    y_test = y_test[index_test[0:n_testing_samples], :]

    # Normalize data
    x_train = x_train / 255.
    x_test = x_test / 255.
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    return x_train, y_train, x_test, y_test


# Fashion-MNIST is a dataset of Zalando's article images consisting of a training set of 60,000 examples and a test set of 10,000 examples.
# Each example is a 28x28 grayscale image, associated with a label from 10 classes.
def load_fashion_mnist_data(n_training_samples, n_testing_samples):

    data = np.load("../data/fashion_mnist.npz")

    index_train = np.arange(data["X_train"].shape[0])
    np.random.shuffle(index_train)

    index_test = np.arange(data["X_test"].shape[0])
    np.random.shuffle(index_test)

    x_train = data["X_train"][index_train[0:n_training_samples], :]
    y_train = data["Y_train"][index_train[0:n_training_samples], :]
    x_test = data["X_test"][index_test[0:n_testing_samples], :]
    y_test = data["Y_test"][index_test[0:n_testing_samples], :]

    # Normalize in 0..1
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.

    return x_train, y_train, x_test, y_test


# The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class.
# There are 50000 training images and 10000 test images.
def load_cifar10_data(n_training_samples, n_testing_samples):

    # read CIFAR10 data
    (x, y), (x_test, y_test) = cifar10.load_data()

    y = np_utils.to_categorical(y, 10)
    y_test = np_utils.to_categorical(y_test, 10)
    x = x.astype('float32')
    x_test = x_test.astype('float32')

    index_train = np.arange(x.shape[0])
    np.random.shuffle(index_train)

    index_test = np.arange(x_test.shape[0])
    np.random.shuffle(index_test)

    x_train = x[index_train[0:n_training_samples], :]
    y_train = y[index_train[0:n_training_samples], :]

    x_test = x_test[index_test[0:n_testing_samples], :]
    y_test = y_test[index_test[0:n_testing_samples], :]

    # Normalize data
    x_train_mean = np.mean(x_train, axis=0)
    x_train_std = np.std(x_train, axis=0)
    x_train = (x_train - x_train_mean) / x_train_std
    x_test = (x_test - x_train_mean) / x_train_std

    x_train = x_train.reshape(-1, 32 * 32 * 3)
    x_test = x_test.reshape(-1, 32 * 32 * 3)

    return x_train, y_train, x_test, y_test


# Not flattened version of CIFAR10
def load_cifar10_data_not_flattened(n_training_samples, n_testing_samples):

    # read CIFAR10 data
    (x, y), (x_test, y_test) = cifar10.load_data()

    y = np_utils.to_categorical(y, 10)
    y_test = np_utils.to_categorical(y_test, 10)
    x = x.astype('float32')
    x_test = x_test.astype('float32')

    index_train = np.arange(x.shape[0])
    np.random.shuffle(index_train)

    index_test = np.arange(x_test.shape[0])
    np.random.shuffle(index_test)

    x_train = x[index_train[0:n_training_samples], :]
    y_train = y[index_train[0:n_training_samples], :]

    x_test = x_test[index_test[0:n_testing_samples], :]
    y_test = y_test[index_test[0:n_testing_samples], :]

    # Normalize data
    x_train_mean = np.mean(x_train, axis=0)
    x_train_std = np.std(x_train, axis=0)
    x_train = (x_train - x_train_mean) / x_train_std
    x_test = (x_test - x_train_mean) / x_train_std

    return x_train, y_train, x_test, y_test


def load_images(curr_dir, label):
    print(f"Loading class {label} images ...")
    class_dir = os.path.join(images_dirs, curr_dir)

    x_train = []
    y_train = []

    # Iterate through the images in the given the folder
    for image_path in os.listdir(class_dir):
        # Create the full input path and read the file
        input_path = os.path.join(class_dir, image_path)
        image = Image.open(input_path)
        x_train.append(np.asarray(image))
        y_train.append(label)

    x_train = np.asarray(x_train).reshape((-1, 64, 64, 3))
    y_train = np.asarray(y_train).flatten()

    print(f"Finished loading for class {label} images ...")
    return x_train, y_train

def load_eurosat__data():
    data = np.load("../data/eurostat.npz", mmap_mode='r')

    return data['X_train'], data['Y_train'], data['X_test'], data['Y_test']

def load_eurosat_parallel():
    class_dirs = os.listdir(images_dirs)

    data = np.array([], dtype='float32').reshape((-1, 64, 64, 3))
    labels = np.array([])

    # Loop through the data folders with training data
    with ProcessPoolExecutor(max_workers=12) as executor:
        results = executor.map(load_images, class_dirs, range(10))
        for i, res in enumerate(results):
            data = np.concatenate((data, res[0]), axis=0)
            labels = np.concatenate((labels, res[1]))

    labels = np_utils.to_categorical(labels, 10)
    data = data.astype('float32')

    index_data = np.arange(data.shape[0])
    np.random.shuffle(data)

    data = data[index_data, :]
    labels = labels[index_data, :]

    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.33, random_state=42)

    # Normalize data
    x_train_mean = np.mean(x_train, axis=0)
    x_train_std = np.std(x_train, axis=0)
    x_train = (x_train - x_train_mean) / x_train_std
    x_test = (x_test - x_train_mean) / x_train_std

    x_train = x_train.reshape(-1, 64 * 64 * 3).astype('float64')
    x_test = x_test.reshape(-1, 64 * 64 * 3).astype('float64')

    return x_train, y_train, x_test, y_test


if __name__ == '__main__':
    x_train, y_train, x_test, y_test = load_eurosat_parallel()
    print(x_train.shape)
    print(y_train.shape)
    print(x_test.shape)
    print(y_test.shape)

    np.savez_compressed('../../data/eurostat.npz', X_train=x_train, Y_train=y_train, X_test=x_test, Y_test=y_test)

    # hf = h5py.File('../../data/CIFAR10/cifar_500K.h5', 'w')
    # hf.create_dataset('x_train', data=x_train, compression='gzip')
    # hf.create_dataset('y_train', data=y_train, compression='gzip')
    # hf.create_dataset('x_test', data=x_test, compression='gzip')
    # hf.create_dataset('y_test', data=y_test, compression='gzip')
    # hf.close()

    # joblib.dump(x_train, '../../data/CIFAR10/cifar_500K_x_train.joblib', compress=3)

    # x_train.dump('../../data/CIFAR10/cifar_500K_x_train.pkl')
