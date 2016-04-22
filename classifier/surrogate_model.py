"""Surrogate model using CV"""

import os
import numpy as np
from common import config, ml_util
from learner import svm
from feature_parser import feature_extractor
from vis import classifier_vis


def generate_figure(cnt, orig, ded, eval_x):
    """Evaluate the model and generate a frame"""
    pred_orig = orig.predict_proba(eval_x)[:, 0]
    pred_ded = ded.predict_proba(eval_x)[:, 0]

    classifier_vis.classifier_vis(
        pred_orig,
        pred_ded,
        out_file='gif_folder/correlation%s.png' % (str(cnt).zfill(4)),
        frame_name=cnt)


def drive():
    """Driver to generate the deduced ML"""
    data_config = config.Section('data')
    root = data_config.get('data-small')
    s_dir = os.path.join(root, 'spam')
    h_dir = os.path.join(root, 'ham')

    assert os.path.exists(s_dir)
    assert os.path.exists(h_dir)

    h_vec, s_vec = feature_extractor.feature_parser(h_dir, s_dir, 1000)
    h_vec, s_vec = np.asmatrix(h_vec), np.asmatrix(s_vec)
    x_data = np.concatenate((h_vec, s_vec), axis=0)
    y_data = np.concatenate(
        (np.zeros((h_vec.shape[0], 1)), np.ones((s_vec.shape[0], 1))),
        axis=0)

    x_bins, y_bins = ml_util.cross_validate(x_data, y_data, 5)

    original_ml_driver = svm.SVM()
    deduced_ml_driver = svm.SVM()

    # Train the blackbox model
    original_ml_driver.train(x_bins[0], y_bins[0])

    # Train surrogate model
    deduced_ml_driver.train(x_bins[1], y_bins[1])

    train_hat = x_bins[1]
    train_labs = y_bins[1]

    # Active Probing starts here - call activeProbing iteratively
    probe_x = np.concatenate(x_bins[2:-1], axis=0)

    # We don't really care about the class label
    # probe_y = np.concatenate(y_bins[2:-1], axis=0)

    for iteration in range(5000):

        prob_orig_ml = original_ml_driver.predict_proba(probe_x)[:, 0]
        prob_ded_ml = deduced_ml_driver.predict_proba(probe_x)[:, 0]
        scores = np.abs(prob_orig_ml - prob_ded_ml)
        worst = np.argmax(scores)

        generate_figure(
            iteration, original_ml_driver, deduced_ml_driver, x_bins[-1])
        if scores[worst] < 0.1:
            break

        target_class = (int(round(prob_orig_ml[worst])) + 1) % 2
        train_hat = np.concatenate((train_hat, probe_x[worst, :]), axis=0)
        train_labs = np.concatenate(
            (train_labs, np.asmatrix(target_class)), axis=0)

        # Retrain

        deduced_ml_driver.train(train_hat, train_labs)

    generate_figure(
        5001,
        original_ml_driver,
        deduced_ml_driver,
        x_bins[-1])


def main():
    """Run the model"""
    # options to be added later
    drive()

if __name__ == "__main__":
    main()
