"""Collection of tests for classifier.surrogate_regression"""

from classifier import surrogate_regression
from learner import svr, svm


def test_surrogate_regression():
    """Test case for running a regression based Surrogate Model"""
    surrogate = svr.SVR(epsilon=0.0001)
    black_box = svm.SVM()

    surrogate_regression.main(black_box, surrogate, 'data-small', None)
