"""Test feature extractor and classifier together"""
import os
from common import config
from feature_parser import feature_extractor
from learner import svm


def test_feat_class():
    """Test both feature extraction and classification"""
    data_config = config.Section('data')
    root = data_config.get('data-small')
    s_dir = os.path.join(root, 'spam')
    h_dir = os.path.join(root, 'ham')

    assert os.path.exists(s_dir)
    assert os.path.exists(h_dir)

    h_vec, s_vec = feature_extractor.feature_parser(h_dir, s_dir, 100)
    print h_vec
    clf = svm.SVM()

    clf.train(
        h_vec + s_vec,
        [1 for _ in range(len(h_vec))] + [0 for _ in range(len(s_vec))])

    pred = clf.predict(h_vec)
    assert sum(pred) > len(pred)/2

    pred = clf.predict(s_vec)
    assert sum(pred) < len(pred)/2
