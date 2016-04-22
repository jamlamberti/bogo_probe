import os
import numpy as np
from common import config, ml_util
from learner import svr, svm
from feature_parser import feature_extractor
from vis import classifier_vis


def generate_figure(cnt, orig, ded, eval_x):
    """Generate a frame"""
    pred_orig = orig.predict_proba(eval_x)[:, 0]
    pred_ded = np.minimum(np.maximum(0.0, ded.predict(eval_x)), 1.0)

    classifier_vis.classifier_vis(
        pred_orig,
        pred_ded,
        out_file='gif_folder/correlation%s.png' % (str(cnt).zfill(4)),
        frame_name=cnt)


def main():
    """Driver to generate the deduced ML"""
    data_config = config.Section('data')

    s_dir = os.path.join(data_config.get('data-small'), 'spam')
    h_dir = os.path.join(data_config.get('data-small'), 'ham')

    assert os.path.exists(s_dir)
    assert os.path.exists(h_dir)

    h_vec, s_vec = feature_extractor.feature_parser(h_dir, s_dir, 250)
    h_vec, s_vec = np.asmatrix(h_vec), np.asmatrix(s_vec)
    x_data = np.concatenate((h_vec, s_vec), axis=0)
    y_data = np.concatenate(
        (np.zeros((h_vec.shape[0], 1)), np.ones((s_vec.shape[0], 1))),
        axis=0)

    x_bins, y_bins = ml_util.cross_validate(x_data, y_data, 5)

    original_ml_driver = svm.SVM()
    deduced_ml_driver = svr.SVR(kernel='rbf', epsilon=0.0001)

    original_ml_driver.train(x_bins[0], y_bins[0])

    probe_x = np.concatenate(x_bins[1:-1], axis=0)
    # DC about labels
    # probe_y = np.concatenate(y_bins[1:-1], axis=0)

    for i in range(probe_x.shape[0], probe_x.shape[0] + 1):
        prob_orig_ml = original_ml_driver.predict_proba(probe_x)[:i, 0]
        deduced_ml_driver.train(probe_x[:i, :], prob_orig_ml)
        generate_figure(i, original_ml_driver, deduced_ml_driver, x_bins[-1])

    generate_figure(
        probe_x.shape[0] + 1,
        original_ml_driver,
        deduced_ml_driver,
        x_bins[-1])


if __name__ == '__main__':
    main()
