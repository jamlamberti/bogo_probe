"""Utilities for Machine Learning"""
import os
import random
import numpy as np
from . import config
from feature_parser import feature_extractor
from vis import classifier_vis


def cross_validate(train_x, train_y, folds=10):
    """Return k-fold CV bins"""
    x_fold = [[] for _ in range(folds)]
    y_fold = [[] for _ in range(folds)]

    for i in range(train_x.shape[0]):
        choice = random.randint(0, folds - 1)
        x_fold[choice].append(np.asmatrix(train_x[i, :]))
        y_fold[choice].append(np.asmatrix(train_y[i, :]))

    for i in range(folds):
        x_fold[i] = np.concatenate(x_fold[i], axis=0)
        y_fold[i] = np.concatenate(y_fold[i], axis=0)

    return x_fold, y_fold


def loss_01(pred1, pred2):
    """Compute 01 Loss"""
    return np.sum(np.abs(pred1 - pred2))


def load_data(data_path, num_feats=500, folds=10):
    """Load data into feature vector and do CV"""
    data_config = config.Section('data')
    root = data_config.get(data_path)
    spam_dir = os.path.join(root, 'spam')
    ham_dir = os.path.join(root, 'ham')

    if not (os.path.exists(spam_dir) or os.path.exists(ham_dir)):
        # Might want to raise custom exception
        raise IOError(
            "Could not find the %s/spam or %s/ham (or maybe both!)" % (
                root,
                root))

    h_vec, s_vec = feature_extractor.feature_parser(
        ham_dir,
        spam_dir,
        num_feats)
    h_vec, s_vec = np.asmatrix(h_vec), np.asmatrix(s_vec)

    x_data = np.concatenate((h_vec, s_vec), axis=0)
    y_data = np.concatenate(
        (np.zeros((h_vec.shape[0], 1)), np.ones((s_vec.shape[0], 1))),
        axis=0)

    return cross_validate(x_data, y_data, folds)


def generate_figure(cnt, pred_orig, pred_ded, out_dir):
    """Generate a frame"""
    if out_dir is None:
        return

    classifier_vis.classifier_vis(
        pred_orig,
        pred_ded,
        out_file=os.path.join(
            out_dir, 'correlation%s.png' % (str(cnt).zfill(4))),
        frame_name=cnt)
