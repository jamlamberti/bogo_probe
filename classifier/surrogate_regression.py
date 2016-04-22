"""Run a regression over the dataset"""

import numpy as np
from common import ml_util


def main(black_box, surrogate, training_data='data-small', out_dir=None):
    """Driver to generate the deduced ML"""
    x_bins, y_bins = ml_util.load_data(training_data)

    # original_ml_driver = svm.SVM()
    # deduced_ml_driver = svr.SVR(kernel='rbf', epsilon=0.0001)

    black_box.train(x_bins[0], y_bins[0])

    probe_x = np.concatenate(x_bins[1:-1], axis=0)

    # DC about labels
    # probe_y = np.concatenate(y_bins[1:-1], axis=0)

    for i in range(probe_x.shape[0], probe_x.shape[0] + 1):
        prob_orig_ml = black_box.predict_proba(probe_x)[:i, 0]
        surrogate.train(probe_x[:i, :], prob_orig_ml)
        ml_util.generate_figure(
            i,
            black_box.predict_proba(x_bins[-1])[:, 0],
            surrogate.predict(x_bins[-1]),
            out_dir)

    ml_util.generate_figure(
        probe_x.shape[0] + 1,
        black_box.predict_proba(x_bins[-1])[:, 0],
        surrogate.predict(x_bins[-1]),
        out_dir)
