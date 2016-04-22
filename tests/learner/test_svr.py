"""A collection of tests for learner.svr"""

import numpy as np
from learner import svr


def test_svr():
    """Test the SVR implementation"""
    learner = svr.SVR()

    # We will generate a perfectly linearly separable
    # dataset
    
    axis = np.linspace(0, 1, num=20)
    xarr, yarr = np.meshgrid(axis, axis)
    data = []
    target = []
    
    for i in range(xarr.shape[0]):
        for j in range(xarr.shape[1]):
            data.append([xarr[i, j], yarr[i, j]])
            target.append(sum(data[-1]))

    data = np.array(data)

    for _ in range(2):
        # Test both with lists and matrices
        learner.train(data, target)
        assert abs(learner.predict([1, 0]) - 1) < 1e-5
        assert abs(learner.predict([0, 1]) - 1) < 1e-5
        target = np.asmatrix(target)
