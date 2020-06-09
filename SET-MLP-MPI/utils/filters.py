import numpy as np
import matplotlib.pyplot as plt
from utils.load_data import *
from keras.datasets import cifar10, mnist
from PIL import Image, ImageFilter
import cv2
import time
import joblib
from skimage.color import rgb2gray


def get_filtered_versions(img, index=0):
    start_time = time.time()
    # Median filter
    median_image = cv2.medianBlur(img, 3)

    # Unsharp filter
    image = Image.fromarray(img.astype('uint8'))
    unsharp_image = image.filter(ImageFilter.UnsharpMask(radius=3, percent=200))
    unsharp_image = np.asarray(unsharp_image)


    gray = rgb2gray(unsharp_image)
    gray_r = gray.reshape(gray.shape[0]*gray.shape[1])
    for i in range(gray_r.shape[0]):
        if gray_r[i] > gray_r.mean():
            gray_r[i] = 3
        elif gray_r[i] > 0.5:
            gray_r[i] = 2
        elif gray_r[i] > 0.25:
            gray_r[i] = 1
        else:
            gray_r[i] = 0
    gray_segmentation = gray_r.reshape(gray.shape[0], gray.shape[1])

    gray = rgb2gray(unsharp_image)
    # Laplacian filter
    laplacian_image = cv2.Laplacian(gray, cv2.CV_64F)
    step_time = time.time() - start_time
    #print(f"Finished image {index}...")
    return median_image, unsharp_image, gray_segmentation, laplacian_image

if __name__ == '__main__':
    # read CIFAR10 data
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    y_train = np_utils.to_categorical(y_train, 10)
    y_test = np_utils.to_categorical(y_test, 10)


    x_train_median = []
    x_train_unsharp = []
    x_train_gray = []
    x_train_laplacian = []

    start_time = time.time()
    with ProcessPoolExecutor(max_workers=12) as executor:
        results = executor.map(get_filtered_versions, x_train, range(x_train.shape[0]))
        for i, res in enumerate(results):
            x_train_median.append(res[0])
            x_train_unsharp.append(res[1])
            x_train_gray.append(res[2])
            x_train_laplacian.append(res[3])
            print(i)

    print(f"Finished filtering for training images ...")
    step_time = time.time() - start_time
    print("\nTotal processing time: ", step_time)

    # for img in x_train:
    #     median_image, unsharp_image, gray_segmentation, laplacian_image = get_filtered_versions(img)
    #     x_train_median.append(median_image)
    #     x_train_unsharp.append(unsharp_image)
    #     x_train_gray.append(gray_segmentation)
    #     x_train_laplacian.append(laplacian_image)


    x_test_median = []
    x_test_unsharp = []
    x_test_gray = []
    x_test_laplacian = []

    start_time = time.time()
    with ProcessPoolExecutor(max_workers=12) as executor:
        results = executor.map(get_filtered_versions, x_test, range(x_test.shape[0]))
        for i, res in enumerate(results):
            x_test_median.append(res[0])
            x_test_unsharp.append(res[1])
            x_test_gray.append(res[2])
            x_test_laplacian.append(res[3])
            print(i)

    # for img in x_test:
    #     median_image, unsharp_image, gray_segmentation, laplacian_image = get_filtered_versions(img)
    #     x_test_median.append(median_image)
    #     x_test_unsharp.append(unsharp_image)
    #     x_test_gray.append(gray_segmentation)
    #     x_test_laplacian.append(laplacian_image)

    print(f"Finished filtering for testing images ...")
    step_time = time.time() - start_time
    print("\nTotal processing time: ", step_time)



    x_train = x_train.reshape(-1, 32 * 32 * 3)
    x_test = x_test.reshape(-1, 32 * 32 * 3)

    # Normalize data
    x_train_mean = np.mean(x_train, axis=0)
    x_train_std = np.std(x_train, axis=0)
    x_train = (x_train - x_train_mean) / x_train_std
    x_test = (x_test - x_train_mean) / x_train_std

    x_train_median = np.asarray(x_train_median).reshape(-1, 32 * 32 * 3)
    x_test_median = np.asarray(x_test_median).reshape(-1, 32 * 32 * 3)
    x_train_unsharp = np.asarray(x_train_unsharp).reshape(-1, 32 * 32 * 3)
    x_test_unsharp = np.asarray(x_test_unsharp).reshape(-1, 32 * 32 * 3)
    x_train_gray = np.asarray(x_train_gray).reshape(-1, 32 * 32)
    x_test_gray = np.asarray(x_test_gray).reshape(-1, 32 * 32)
    x_train_laplacian = np.asarray(x_train_laplacian).reshape(-1, 32 * 32)
    x_test_laplacian = np.asarray(x_test_laplacian).reshape(-1, 32 * 32)

    x_train_median_mean = np.mean(x_train_median, axis=0)
    x_train_median_std = np.std(x_train_median, axis=0)
    x_train_median = (x_train_median - x_train_median_mean) / x_train_median_std
    x_test_median = (x_test_median - x_train_median_mean) / x_train_median_std

    x_train_unsharp_mean = np.mean(x_train_unsharp, axis=0)
    x_train_unsharp_std = np.std(x_train_unsharp, axis=0)
    x_train_unsharp = (x_train_unsharp - x_train_unsharp_mean) / x_train_unsharp_std
    x_test_unsharp = (x_test_unsharp - x_train_unsharp_mean) / x_train_unsharp_std

    x_train_gray_mean = np.mean(x_train_gray, axis=0)
    x_train_gray_std = np.std(x_train_gray, axis=0)
    x_train_gray = (x_train_gray - x_train_gray_mean) / x_train_gray_std
    x_test_gray = (x_test_gray - x_train_gray_mean) / x_train_gray_std

    x_train_laplacian_mean = np.mean(x_train_laplacian, axis=0)
    x_train_laplacian_std = np.std(x_train_laplacian, axis=0)
    x_train_laplacian = (x_train_laplacian - x_train_laplacian_mean) / x_train_laplacian_std
    x_test_laplacian = (x_test_laplacian - x_train_laplacian_mean) / x_train_laplacian_std

    print(x_train.shape)
    print(y_train.shape)
    print(x_train_median.shape)
    print(x_train_unsharp.shape)
    print(x_train_gray.shape)
    print(x_train_laplacian.shape)

    print(x_test.shape)
    print(y_test.shape)
    print(x_test_median.shape)
    print(x_test_unsharp.shape)
    print(x_test_gray.shape)
    print(x_test_laplacian.shape)

    x_train_features = np.hstack((x_train, x_train_median, x_train_gray, x_train_laplacian))
    x_test_features = np.hstack((x_test, x_test_median, x_test_gray, x_test_laplacian))

    print(x_train_features.shape)
    print(x_test_features.shape)

    joblib.dump(x_train_features, '../../data/CIFAR10/x_train_features.joblib', compress=3)
    joblib.dump(x_test_features, '../../data/CIFAR10/x_test_features.joblib', compress=3)
    joblib.dump(y_train, '../../data/CIFAR10/y_train_features.joblib', compress=3)
    joblib.dump(y_test, '../../data/CIFAR10/y_test_features.joblib', compress=3)
