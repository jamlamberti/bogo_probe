"""A collection of tests for histogram.py"""

import os
import random
from vis.histogram import histogram_analysis


def test_histogram():
    """A smoke test"""
    x_range = random.sample(range(1000), 100)
    xbins = [0, len(xrange)]
    histogram_analysis(x_range, xbins, 'tmp.png')
    assert os.path.exists('tmp.png')
    os.remove('tmp.png')
