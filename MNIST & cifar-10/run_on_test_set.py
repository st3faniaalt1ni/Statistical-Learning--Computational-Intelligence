import pickle
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn import decomposition
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.utils import check_random_state
from tensorflow.keras.datasets import cifar10
import tensorflow as tf

if __name__ == '__main__':

    # DATASET = 'MNIST'
    DATASET = 'CIFAR10'

    SEED = 0
    np.random.seed(SEED)

    if DATASET == 'MNIST':

        LABELS = {
            0: 'Even number',
            1: 'Odd number'
        }

        n_components = 87
        best_classifier = pickle.load(open(f'{DATASET}_best_classifier.sav', 'rb'))

        X, y = fetch_openml('mnist_784', version=1, return_X_y=True, data_home='./data', cache=True)
        X = X / 255.  # normalize to [0, 1]
        y = y.astype('int') % 2  # odd = 1, even = 0
        # y = np.array(list(map(lambda x: (int(x) % 2 == 0) * 1, y))).astype(np.str)  # remake labels for odd/even task

        dataset_size = len(y)
        train_samples = int(.6 * dataset_size)
        test_samples = dataset_size - train_samples

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=train_samples, test_size=test_samples)

        # PCA for dimension reduction
        pca = decomposition.PCA(n_components=n_components)  # retain > 90% (0.90) variance
        X_PCA = pca.fit_transform(X_test)
        random_state = check_random_state(0)

        y_pred = best_classifier.predict(X_PCA)
        error_idx = list(np.where(y_test != y_pred)[0])
        correct_idx = list(np.where(y_test == y_pred)[0])

        # plots 2 false and 2 correct predictions
        fig, axs = plt.subplots(2, 2)

        axs[0, 0].imshow(X_test[error_idx[0]].reshape((28, 28)) * 255)
        axs[0, 0].set_title(f"Prediction: {LABELS[y_pred[error_idx[0]]]}\n"
                            f"GT Label: {LABELS[y_test[error_idx[0]]]}",
                            fontdict={'fontsize': 8, 'fontweight': 'medium'})
        axs[0, 0].axis('off')

        axs[0, 1].imshow(X_test[error_idx[1]].reshape((28, 28)) * 255)
        axs[0, 1].set_title(f"Prediction: {LABELS[y_pred[error_idx[1]]]}\n"
                            f"GT Label: {LABELS[y_test[error_idx[1]]]}",
                            fontdict={'fontsize': 8, 'fontweight': 'medium'})
        axs[0, 1].axis('off')

        axs[1, 0].imshow(X_test[correct_idx[0]].reshape((28, 28)) * 255)
        axs[1, 0].set_title(f"Prediction: {LABELS[y_pred[correct_idx[0]]]}\n"
                            f"GT Label: {LABELS[y_test[correct_idx[0]]]}",
                            fontdict={'fontsize': 8, 'fontweight': 'medium'})
        axs[1, 0].axis('off')

        axs[1, 1].imshow(X_test[correct_idx[1]].reshape((28, 28)) * 255)
        axs[1, 1].set_title(f"Prediction: {LABELS[y_pred[correct_idx[1]]]}\n"
                            f"GT Label: {LABELS[y_test[correct_idx[1]]]}",
                            fontdict={'fontsize': 8, 'fontweight': 'medium'})
        axs[1, 1].axis('off')

        fig.savefig(f'{DATASET}_samples.png', dpi=fig.dpi)

    elif DATASET == 'CIFAR10':

        LABELS = {
            0: 'airplane',
            1: 'automobile',
            2: 'bird',
            3: 'cat',
            4: 'deer',
            5: 'dog',
            6: 'frog',
            7: 'horse',
            8: 'ship',
            9: 'truck',

        }

        n_components = 76
        best_classifier = pickle.load(open(f'{DATASET}_best_classifier.sav', 'rb'))

        (_, _), (X_test, y_test) = cifar10.load_data()

        X_test_rgb = X_test
        X_test_rgb = np.squeeze(np.array(X_test_rgb))

        X_test = X_test.astype(np.float)
        X_test = tf.image.rgb_to_grayscale(X_test)
        X_test = X_test / 255.

        X_test = np.squeeze(np.array(X_test))
        num_test_sampels, h, w = X_test.shape
        X_test = X_test.reshape((num_test_sampels, h * w))

        y_test = np.squeeze(y_test)

        # PCA for dimension reduction
        pca = decomposition.PCA(n_components=n_components)  # retain > 90% (0.90) variance
        X_PCA = pca.fit_transform(X_test)
        random_state = check_random_state(0)

        y_pred = best_classifier.predict(X_PCA)

        error_idx = list(np.where(y_test != y_pred)[0])
        correct_idx = list(np.where(y_test == y_pred)[0])

        # plots 2 false and 2 correct predictions
        fig, axs = plt.subplots(2, 2)

        # axs[0, 0].imshow(X_test_rgb[error_idx[0]].reshape((32, 32)) * 255)
        axs[0, 0].imshow(X_test_rgb[error_idx[0]] * 255)
        axs[0, 0].set_title(f"Prediction: {LABELS[y_pred[error_idx[0]]]}\n"
                            f"GT Label: {LABELS[y_test[error_idx[0]]]}",
                            fontdict={'fontsize': 8, 'fontweight': 'medium'})
        axs[0, 0].axis('off')

        # axs[0, 1].imshow(X_test_rgb[error_idx[1]].reshape((32, 32)) * 255)
        axs[0, 1].imshow(X_test_rgb[error_idx[1]] * 255)
        axs[0, 1].set_title(f"Prediction: {LABELS[y_pred[error_idx[1]]]}\n"
                            f"GT Label: {LABELS[y_test[error_idx[1]]]}",
                            fontdict={'fontsize': 8, 'fontweight': 'medium'})
        axs[0, 1].axis('off')

        axs[1, 0].imshow(X_test_rgb[correct_idx[0]] * 255)
        axs[1, 0].set_title(f"Prediction: {LABELS[y_pred[correct_idx[0]]]}\n"
                            f"GT Label: {LABELS[y_test[correct_idx[0]]]}",
                            fontdict={'fontsize': 8, 'fontweight': 'medium'})
        axs[1, 0].axis('off')

        axs[1, 1].imshow(X_test_rgb[correct_idx[1]] * 255)
        axs[1, 1].set_title(f"Prediction: {LABELS[y_pred[correct_idx[1]]]}\n"
                            f"GT Label: {LABELS[y_test[correct_idx[1]]]}",
                            fontdict={'fontsize': 8, 'fontweight': 'medium'})
        axs[1, 1].axis('off')

        fig.savefig(f'{DATASET}_samples.png', dpi=fig.dpi)
    # plt.show()

    else:
        raise NotImplemented

    X_test = np.load(f'{DATASET}_X_test.npy')
    y_test = np.load(f'{DATASET}_y_test.npy')

    y_pred = best_classifier.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    # test_f1 = f1_score(y_test, y_pred)
    conf_mat = confusion_matrix(y_test, y_pred)

    print(f'SVM Trained Classifier Accuracy: {test_acc}')
    # print(f'SVM Trained Classifier F1: {test_f1}')
    print(f'Confusion Matrix: \n{conf_mat}', )
