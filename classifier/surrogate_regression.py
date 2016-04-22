"""Run a regression over the dataset"""

import numpy as np
from common import ml_util
from learner import svr, svm
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
    x_bins, y_bins = ml_util.load_data('data_small')

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
