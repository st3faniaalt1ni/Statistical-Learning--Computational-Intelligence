import numpy as np
import tensorflow as tf
from sklearn import decomposition
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.utils import check_random_state
from tensorflow.keras.datasets import cifar10



def prepare_data(dataset_name='MNIST', apply_pca=True, train_samples=None, test_samples=None):
    if dataset_name == 'MNIST':

        X, y = fetch_openml('mnist_784', version=1, return_X_y=True, data_home='./data', cache=True)
        X = X / 255.  # normalize to [0, 1]
        y = y.astype('int') % 2  # odd = 1, even = 0

        dataset_size = len(y)
        if train_samples is None:
            train_samples = int(.6 * dataset_size)
            test_samples = dataset_size - train_samples
        else:
            train_samples=train_samples
            test_samples=test_samples

        if apply_pca:
            # PCA for dimension reduction
            pca = decomposition.PCA(n_components=.90, svd_solver='full')  # retain > 90% (0.90) variance
            X = pca.fit_transform(X)

        # fixed random seed for identical shuffling (permutation)
        random_state = check_random_state(0)  # to reproduce results (same randomness)
        permutation = random_state.permutation(X.shape[0])
        X = X[permutation]
        y = y[permutation]

        # split data into training/testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=train_samples, test_size=test_samples)


    elif dataset_name == 'CIFAR10':

        (X_train, y_train), (X_test, y_test) = cifar10.load_data()
        X_train = X_train.astype(np.float)
        X_train = tf.image.rgb_to_grayscale(X_train)
        X_train = X_train / 255.

        X_train = np.squeeze(np.array(X_train))
        num_train_sampels, h, w = X_train.shape
        X_train = X_train.reshape((num_train_sampels, h * w))

        X_test = X_test.astype(np.float)
        X_test = tf.image.rgb_to_grayscale(X_test)
        X_test = X_test / 255.

        X_test = np.squeeze(np.array(X_test))
        num_test_sampels, h, w = X_test.shape
        X_test = X_test.reshape((num_test_sampels, h * w))

        y_train = np.squeeze(y_train)
        y_test = np.squeeze(y_test)

        if apply_pca:
            # PCA for dimension reduction
            pca = decomposition.PCA(n_components=.90, svd_solver='full')  # retain > 90% (0.90) variance
            X_train_PCA = pca.fit_transform(X_train)
            _, num_components = X_train_PCA.shape  # num_components for retaining > 90% variance
            X_train = X_train_PCA

            pca = decomposition.PCA(n_components=num_components)
            X_test_PCA = pca.fit_transform(X_test)
            X_test = X_test_PCA

        if train_samples is not None:
            X_train = X_train[0:train_samples, :]
            y_train = y_train[0:train_samples,]
            X_test = X_test[0:test_samples, :]
            y_test = y_test[0:test_samples,]
    else:
        raise NotImplemented('Implemented datasets include \'MNIST\',  \'IRIS\' and \'CIFAR-10\'.')

    return (X_train, X_test, y_train, y_test)


if __name__ == '__main__':
    pass
