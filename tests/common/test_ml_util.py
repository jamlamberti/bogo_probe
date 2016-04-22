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


def test_loss_01():
    """Test case for 01 Loss"""
    pred1 = np.array([1, 1, 1, 0, 0, 1, 0])
    pred2 = np.array([1, 1, 0, 0, 1, 1, 1])

    assert ml_util.loss_01(pred1, pred2) == 3
    assert ml_util.loss_01(pred2, pred1) == 3
