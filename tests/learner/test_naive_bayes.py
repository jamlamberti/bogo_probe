"""A collection of tests for learner.naive_bayes"""

import numpy as np
from learner import naive_bayes


def test_naive_bayes():
    """Test the Naive Bayes Implementation"""
    learner = naive_bayes.NaiveBayes()

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
    learner.train(data, target)
    assert learner.predict([1, 0]) == 0
    assert learner.predict([0, 1]) == 1
    assert all([abs(1. - i) < 1e-5
                for i in np.sum(learner.predict_proba(data), axis=1)])
