"""A collection of tests for learner.svm"""

import numpy as np
from learner import svm


def test_svm():
    """Test the SVM implementation"""
    learner = svm.SVM()

    # We will generate a perfectly linearly separable
    # dataset
    axis = np.linspace(0, 1, num=20)
    xarr, yarr = np.meshgrid(axis, axis)
    data = []
    target = []
    for i in range(xarr.shape[0]):
        for j in range(xarr.shape[1]):
            data.append([xarr[i, j], yarr[i, j]])
            target.append(1 if xarr[i, j] <= yarr[i, j] else 0)
    data = np.array(data)
    for _ in range(2):
        # Test both with lists and matrices
        learner.train(data, target)
        assert learner.predict([1, 0]) == 0
        assert learner.predict([0, 1]) == 1
        assert all([abs(1. - i) < 1e-5
                    for i in np.sum(learner.predict_proba(data), axis=1)])
        target = np.asmatrix(target)
