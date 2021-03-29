import datetime as dt
from loguru import logger
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import random
from sklearn import svm, datasets, pipeline
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ShuffleSplit

from data import prepare_data

if __name__ == '__main__':
    DATASET = 'CIFAR10'
    # DATASET = 'MNIST'

    SEED = 0
    np.random.seed(SEED)  # to reproduce results (same randomness)

    logger.add(f'{DATASET}.log')  # log results (in .log)

    X_train, X_test, y_train, y_test = prepare_data(DATASET)

    # GridSearch for hyperparameter tuning (run SVM with all combinations of params)
    models = [
        ('SVM', svm.SVC())
    ]
    pipeline = pipeline.Pipeline(models)

    parameters = {
        'SVM__kernel': ['linear', 'poly', 'rbf'],
        'SVM__C': [0.1, 0.5],
        'SVM__gamma': [0.5, 0.1, 0.01]}

    cv = ShuffleSplit(n_splits=1, test_size=0.2, random_state=0)  # cross-validation setup (1 split, 20% val set)
    grid = GridSearchCV(pipeline, param_grid=parameters, scoring='f1_micro', verbose=0, cv=cv)

    # train all SVMs
    grid.fit(X_train, y_train)

    # save GridSearchCV results in .csv and log best params
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', -1)

    results = pd.DataFrame(grid.cv_results_)
    logger.info(f'\n{results}')
    logger.info('')
    logger.info(f'\nBest parameters: {grid.best_params_}')
    results.to_csv(path_or_buf=f'{DATASET}_results.csv')

    # save best classifier
    best_classifier = grid.best_estimator_
    filename = f'{DATASET}_best_classifier.sav'
    pickle.dump(best_classifier, open(filename, 'wb'))

    # save X_test, y_test as numpy files
    filename = f'{DATASET}_X_test.npy'
    np.save(filename, X_test)

    filename = f'{DATASET}_y_test.npy'
    np.save(filename, y_test)
