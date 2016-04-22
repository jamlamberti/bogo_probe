"""Utilities for Machine Learning"""
import random
import numpy as np


def cross_validate(train_x, train_y, folds=10):
    """Return k-fold CV bins"""
    x_fold = [[] for _ in range(folds)]
    y_fold = [[] for _ in range(folds)]

    for i in range(train_x.shape[0]):
        choice = random.randint(0, folds-1)
        x_fold[choice].append(np.asmatrix(train_x[i, :]))
        y_fold[choice].append(np.asmatrix(train_y[i, :]))

    for i in range(folds):
        x_fold[i] = np.concatenate(x_fold[i], axis=0)
        y_fold[i] = np.concatenate(y_fold[i], axis=0)

    return x_fold, y_fold

def loss_01(pred1, pred2):
    """Compute 01 Loss"""
    return np.sum(np.abs(pred1-pred2))
