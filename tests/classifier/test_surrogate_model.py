"""Collection of tests for classifier.surrogate_model"""

from classifier import surrogate_model
from learner import svm


def test_surrogate_regression():
    """Test case for running a classifier based Surrogate Model"""
    surrogate = svm.SVM()
    black_box = svm.SVM()

    surrogate_model.main(
        black_box,
        surrogate,
        training_data='data-small',
        out_dir=None,
        threshold=0.2,
        iterations=100)
