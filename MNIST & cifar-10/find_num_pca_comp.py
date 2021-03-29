import numpy as np
import matplotlib.pyplot as plt
from sklearn import decomposition
from sklearn.datasets import fetch_openml
from tensorflow.keras.datasets import cifar10
import tensorflow as tf

DATASET = 'CIFAR10'

if DATASET == 'MNIST':
    # MNIST
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True, data_home='./data', cache=True)
    X = X / 255.
    y = np.array(list(map(lambda x: (int(x) % 2 == 0) * 1, y))).astype(np.str)  # remake labels for odd/even task

    pca = decomposition.PCA()
    pca.n_components = 784
    pca_data = pca.fit_transform(X)

    percentage_var_explained = pca.explained_variance_ / np.sum(pca.explained_variance_)
    cum_var_explained = np.cumsum(percentage_var_explained)

elif DATASET == 'CIFAR10':
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    X_train = X_train.astype(np.float)
    X_train = tf.image.rgb_to_grayscale(X_train)
    X_train = X_train / 255.

    X_train = np.squeeze(np.array(X_train))
    num_train_sampels, h, w = X_train.shape
    X_train = X_train.reshape((num_train_sampels, h * w))

    pca = decomposition.PCA()
    pca.n_components = 1024
    pca_data = pca.fit_transform(X_train)

    percentage_var_explained = pca.explained_variance_ / np.sum(pca.explained_variance_)
    cum_var_explained = np.cumsum(percentage_var_explained)

# Plot the PCA spectrum
plt.figure(1, figsize=(6, 4))

plt.clf()
plt.plot(cum_var_explained, linewidth=2)
plt.axis('tight')
plt.grid()
plt.xlabel('n_components')
plt.ylabel('Cumulative_explained_variance')
plt.show()
