"""A collection of tests for vis.classifier_vis"""
import os

from vis.classifier_vis import classifier_vis


def test_classifier_vis():
    """A simple smoke test"""
    import numpy as np
    points = 100
    prob = np.random.uniform(0, 1, (points))
    expected = np.minimum(
        np.maximum(np.random.normal(0, 0.05, (points)) + prob, 0.0), 1.0)
    classifier_vis(prob, expected, out_file='tmp.png')
    assert os.path.exists('tmp.png')
    os.remove('tmp.png')
