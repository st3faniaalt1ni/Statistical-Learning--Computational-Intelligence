# -*- coding: utf-8 -*-
"""kpca_lda.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Qw5nknsIt5YSfZJ6LuNDx_dK5viO_Kx1
"""

!pip install loguru

from sklearn.decomposition import  KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from data import prepare_data
from loguru import logger
import numpy as np
from sklearn import neighbors
from sklearn.model_selection import cross_validate
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.neighbors.nearest_centroid import NearestCentroid

DATASET = 'CIFAR10'
# DATASET = 'MNIST'

SEED = 0
np.random.seed(SEED)  # to reproduce results (same randomness)
logger.add(f'{DATASET}_kpca_lda.log')  # log results (in .log)

if DATASET == 'MNIST':
    X_train, X_test, y_train, y_test = prepare_data(DATASET, apply_pca=False, train_samples=17000, test_samples=10000)
else:
    X_train, X_test, y_train, y_test = prepare_data(DATASET, apply_pca=False, train_samples=17000, test_samples=10000)

if DATASET == 'MNIST':
    kpca = KernelPCA(kernel="poly",n_components=87 , gamma=1, n_jobs=1) # MNIST
else:
    kpca = KernelPCA(kernel="rbf", degree=3, n_components=75 , gamma=.5, n_jobs=1) # CIFAR-10
    
X_kpca = kpca.fit_transform(X_train)
X_test = kpca.transform(X_test)
print(kpca)
print(X_kpca.shape)
DATASET = 'CIFAR10'
# DATASET = 'MNIST'

SEED = 0
np.random.seed(SEED)  # to reproduce results (same randomness)
logger.add(f'{DATASET}_kpca_lda.log')  # log results (in .log)

if DATASET == 'MNIST':
    X_train, X_test, y_train, y_test = prepare_data(DATASET, apply_pca=False, train_samples=17000, test_samples=10000)
else:
    X_train, X_test, y_train, y_test = prepare_data(DATASET, apply_pca=False, train_samples=17000, test_samples=10000)

if DATASET == 'MNIST':
    kpca = KernelPCA(kernel="poly",n_components=87 , gamma=1, n_jobs=1) # MNIST
else:
    kpca = KernelPCA(kernel="rbf", degree=3, n_components=75 , gamma=.5, n_jobs=1) # CIFAR-10
    
X_kpca = kpca.fit_transform(X_train)
X_test = kpca.transform(X_test)
print(kpca)
print(X_kpca.shape)

lda = LDA()
print (lda)

X_lda = lda.fit_transform(X_kpca,y_train)
X_test = lda.transform(X_test)
print(X_lda.shape)

#kNN classification
clf = neighbors.KNeighborsClassifier(n_neighbors=2)
clf.fit(X_lda, y_train)
print (clf)

# training metrics
y_tr_pred = clf.predict(X_lda)
train_acc = accuracy_score(y_tr_pred, y_train)
train_conf_mat = confusion_matrix(y_train, y_tr_pred)

print(f'kNN Training Accuracy: {train_acc}')
print(f'Training Confusion Matrix: \n{train_conf_mat}', )

# testing metrics
y_pred = clf.predict(X_test)
test_acc = accuracy_score(y_test, y_pred)
conf_mat = confusion_matrix(y_test, y_pred)

print(f'kNN Trained Classifier (Testing) Accuracy: {test_acc}')
print(f'Confusion Matrix: \n{conf_mat}', )

#Nearest Centroid classification
classifier = NearestCentroid(shrink_threshold=.5)
classifier.fit(X_lda, y_train)
NearestCentroid(metric='euclidean', shrink_threshold=None)
print (classifier)

# training metrics
y_tr_pred = classifier.predict(X_lda)
train_acc = accuracy_score(y_tr_pred, y_train)
train_conf_mat = confusion_matrix(y_train, y_tr_pred)

print(f'Nearest Centroid Training Accuracy: {train_acc}')
print(f'Training Confusion Matrix: \n{train_conf_mat}', )

# testing metrics
y_pred = classifier.predict(X_test)
test_acc = accuracy_score(y_test, y_pred)
conf_mat = confusion_matrix(y_test, y_pred)

print(f'Nearest Centroid Trained Classifier (Testing) Accuracy: {test_acc}')
print(f'Confusion Matrix: \n{conf_mat}', )