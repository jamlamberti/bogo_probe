"""Surrogate model using CV"""

import numpy as np
from common import ml_util


def main(
        black_box,
        surrogate,
        training_data='data-small',
        out_dir=None,
        threshold=0.1,
        iterations=500):
    """Driver to generate the deduced ML"""
    x_bins, y_bins = ml_util.load_data(training_data, 500, 5)

    # Train the blackbox model
    black_box.train(x_bins[0], y_bins[0])

    # Train surrogate model
    surrogate.train(x_bins[1], y_bins[1])

    train_hat = x_bins[1]
    train_labs = y_bins[1]

    # Active Probing starts here - call activeProbing iteratively
    probe_x = np.concatenate(x_bins[2:-1], axis=0)

    # We don't really care about the class label
    # probe_y = np.concatenate(y_bins[2:-1], axis=0)

    prob_orig_ml = black_box.predict_proba(probe_x)[:, 0]

    for iteration in range(iterations):
        prob_ded_ml = surrogate.predict_proba(probe_x)[:, 0]
        scores = np.abs(prob_orig_ml - prob_ded_ml)
        # Indexes greater than threshhold
        indexes = sorted(
            np.where(scores[:] > threshold)[0],
            key=lambda x, y=scores: y[x],
            reverse=True)
        updated = False
        for index in indexes:
            # Try to add it and see if we improve
            train_hat_p = np.concatenate(
                (train_hat, probe_x[index, :]), axis=0)
            train_labs_p = np.concatenate((
                train_labs,
                np.asmatrix(
                    (int(round(prob_orig_ml[index])) + 1) % 2)), axis=0)

            surrogate.train(train_hat_p, train_labs_p)

            updated_scores = np.abs(
                prob_orig_ml - surrogate.predict_proba(probe_x)[:, 0])

            if np.sum(scores) > np.sum(updated_scores):
                # Commit the changes
                train_hat = train_hat_p
                train_labs = train_labs_p
                updated = True
                break

        if not updated:
            break

        # worst = np.argmax(scores)

        ml_util.generate_figure(
            iteration,
            black_box.predict_proba(x_bins[-1])[:, 0],
            surrogate.predict_proba(x_bins[-1])[:, 0],
            out_dir)

        # if scores[worst] < threshold:
        #    break

        # target_class = (int(round(prob_orig_ml[worst])) + 1) % 2
        # train_hat = np.concatenate((train_hat, probe_x[worst, :]), axis=0)

        # train_labs = np.concatenate(
        #    (train_labs, np.asmatrix(target_class)), axis=0)

        # Retrain
        # surrogate.train(train_hat, train_labs)

    ml_util.generate_figure(
        iterations,
        black_box.predict_proba(x_bins[-1])[:, 0],
        surrogate.predict_proba(x_bins[-1])[:, 0],
        out_dir)
