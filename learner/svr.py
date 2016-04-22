"""An SVR implementation"""

import numpy as np
import sklearn.svm

from .learner import Learner

# Currently just a wrapper around SKLearn
# I think we might want to look into
# online variants (e.g. passive-aggressive algos)

# A generic implementation with multiple kernels
# would be nice!!


class SVR(Learner):

    """SVM for Regression Wrapper"""

    def __init__(
            self,
            kernel='linear',
            degree=3,
            epsilon=0.01,
            # gamma='auto',
            coef0=0.0,
    ):
        super(SVR, self).__init__()

        self.classifier = sklearn.svm.SVR(
            kernel=kernel,
            epsilon=epsilon,
            degree=degree,
            # gamma=gamma,
            coef0=coef0,
        )
        self.log.debug("Initialized an SVR classifier with:")
        self.log.debug(
            '    kernel=%s, degree=%d, coef0=%0.3f',
            kernel,
            degree,
            # gamma,
            coef0)

    def train(self, train_x, train_y):
        """
        Train the SVM classifier
        """
        self.log.info("Training SVR classifier")
        if len(train_y.shape) == 2:
            self.classifier.fit(train_x, np.asarray(train_y).reshape(-1))
        else:
            self.classifier.fit(train_x, train_y)
        self.log.info("Done training SVR classifier")

    def predict(self, test_x):
        """
        Return predicted class labels
        """
        self.log.info("Computing SVR predictions")
        return self.classifier.predict(test_x)

    def predict_proba(self, test_x):
        """
        Return predicted probabilities from SVR classifier
        """

        self.log.info("Computing SVR probabilities")
        return self.classifier.predict_proba(test_x)
