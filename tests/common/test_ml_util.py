"""Collection of tests for common/ml_util"""
import numpy as np
from common import ml_util


def test_cross_validate():
    """Unit test for cv code"""
    size = 1000
    folds = 5
    train_x = np.random.standard_normal((size, 5))
    train_y = np.random.randint(0, 2, (size, 1))
    x_bins, y_bins = ml_util.cross_validate(train_x, train_y, folds)

    assert size == sum([x.shape[0] for x in x_bins])
    assert size == sum([y.shape[0] for y in y_bins])
    assert len(y_bins) == folds
    assert len(x_bins) == folds


def test_evalutation():
    """Test case for 01 Loss"""
    pred1 = np.array([1, 1, 1, 0, 0, 1, 0])
    pred2 = np.array([1, 1, 0, 0, 1, 1, 1])

    assert ml_util.loss_01(pred1, pred2) == 3
    assert ml_util.loss_01(pred2, pred1) == 3

    assert ml_util.compute_accuracy(pred1, pred2) == 4.0/7
    assert ml_util.compute_accuracy(pred2, pred1) == 4.0/7

    assert ml_util.compute_precision(pred1, pred2) == 3.0/(3+2)
    assert ml_util.compute_precision(pred2, pred1) == 3.0/(3+1)

    assert ml_util.compute_recall(pred1, pred2) == 3.0/(3+1)
    assert ml_util.compute_recall(pred2, pred1) == 3.0/(3+2)
    fscore = 2*(9.0/((3+1)*(3+2)))/(3.0/(3+1) + 3.0/(3+2))
    assert abs(ml_util.compute_fscore(pred1, pred2) - fscore) < 1e-4
    assert abs(ml_util.compute_fscore(pred2, pred1) - fscore) < 1e-4


def test_load_data():
    """Test the load data and CV methods"""
    num_feats = 10
    folds = 5
    x_bins, y_bins = ml_util.load_data(
        "data-small",
        num_feats=num_feats,
        folds=folds)

    assert len(y_bins) == folds
    assert len(x_bins) == folds
    assert all([x.shape[1] == num_feats for x in x_bins])
    assert all([x_bins[i].shape[0] == y_bins[i].shape[0]
                for i in range(folds)])
