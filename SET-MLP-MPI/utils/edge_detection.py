import numpy as np
import matplotlib.pyplot as plt
from utils.load_data import *
from keras.datasets import cifar10, mnist
from PIL import Image, ImageFilter
import cv2

# define the vertical filter
vertical_filter = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]

# define the horizontal filter
horizontal_filter = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]

# read CIFAR10 data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# x_train_mean = np.mean(x_train, axis=0)
# x_train_std = np.std(x_train, axis=0)
# x_train = (x_train - x_train_mean) / x_train_std
# x_test = (x_test - x_train_mean) / x_train_std

# y_train = np_utils.to_categorical(y_train, 10)
# y_test = np_utils.to_categorical(y_test, 10)

i=250
img = x_train[i]
print(y_train[i])
plt.imshow(img)
plt.show()
# get the dimensions of the image
n, m, d = img.shape

# initialize the edges image
edges_img = img.copy()

# loop over all pixels in the image
for row in range(3, n - 2):
    for col in range(3, m - 2):
        # create little local 3x3 box
        local_pixels = img[row - 1:row + 2, col - 1:col + 2, 0]

        # apply the vertical filter
        vertical_transformed_pixels = vertical_filter * local_pixels
        # remap the vertical score
        vertical_score = vertical_transformed_pixels.sum() / 4

        # apply the horizontal filter
        horizontal_transformed_pixels = horizontal_filter * local_pixels
        # remap the horizontal score
        horizontal_score = horizontal_transformed_pixels.sum() / 4

        # combine the horizontal and vertical scores into a total edge score
        edge_score = (vertical_score ** 2 + horizontal_score ** 2) ** .5

        # insert this edge score into the edges image
        edges_img[row, col] = [edge_score] * 3

# remap the values in the 0-1 range in case they went out of bounds
edges_img = edges_img / edges_img.max()

plt.imshow(edges_img)
plt.show()

# Mean filter
mean_image = cv2.blur(img, (3, 3))
fig, ax = plt.subplots(1,2)
ax[0].imshow(img)
ax[1].imshow(mean_image)
plt.show()

# Gaussian filter
gaussian_image = cv2.GaussianBlur(img, (3, 3), 0)
fig, ax = plt.subplots(1,2)
ax[0].imshow(img)
ax[1].imshow(gaussian_image)
plt.show()

# Median filter
median_image = cv2.medianBlur(img, 3)
fig, ax = plt.subplots(1,2)
ax[0].imshow(img)
ax[1].imshow(median_image)
plt.show()

# Laplacian filter
laplacian_image = cv2.Laplacian(gaussian_image, cv2.CV_64F)
fig, ax = plt.subplots(1,2)
ax[0].imshow(img)
ax[1].imshow(laplacian_image)
plt.show()

#sharpen
image = Image.fromarray(img.astype('uint8'))
new_image = image.filter(ImageFilter.UnsharpMask(radius=3, percent=200))
fig, ax = plt.subplots(1,2)
ax[0].imshow(img)
ax[1].imshow(new_image)
plt.show()


new_image = np.asarray(new_image)
from skimage.color import rgb2gray
gray = rgb2gray(new_image)
plt.imshow(gray, cmap='gray')
plt.show()
gray_r = gray.reshape(gray.shape[0]*gray.shape[1])
for i in range(gray_r.shape[0]):
    if gray_r[i] > gray_r.mean():
        gray_r[i] = 1
    else:
        gray_r[i] = 0
gray_r = gray_r.reshape(gray.shape[0],gray.shape[1])
plt.imshow(gray_r, cmap='gray')
plt.show()


gray = rgb2gray(new_image)
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
gray = gray_r.reshape(gray.shape[0],gray.shape[1])
plt.imshow(gray, cmap='gray')
plt.show()


gray = rgb2gray(new_image)
# Laplacian filter
laplacian_image = cv2.Laplacian(gray, cv2.CV_64F)
fig, ax = plt.subplots(1,2)
ax[0].imshow(img)
ax[1].imshow(laplacian_image)
plt.show()