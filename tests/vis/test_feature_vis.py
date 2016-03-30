"""A collection of tests for vis.feature_vis"""
import os
import numpy as np

from vis.feature_vis import feature_vis


def test_feature_vis():
    """A smoke test"""
    data = np.random.uniform(0, 10, (50, 50))
    feature_vis(data, 'tmp.png')
    assert os.path.exists('tmp.png')
    os.remove('tmp.png')
