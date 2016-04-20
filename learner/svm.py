"""An SVM implementation"""

import numpy as np
import sklearn.svm

from .learner import Learner

# Currently just a wrapper around SKLearn
# I think we might want to look into
# online variants (e.g. passive-aggressive algos)

# A generic implementation with multiple kernels
# would be nice!!


class SVM(Learner):

    """SVM Wrapper"""

    def __init__(
            self,
            kernel='linear',
            degree=3,
            # gamma='auto',
            coef0=0.0,
    ):
        super(SVM, self).__init__()

        self.classifier = sklearn.svm.SVC(
            kernel=kernel,
            degree=degree,
            # gamma=gamma,
            probability=True,
            coef0=coef0,
        )
        self.log.debug("Initialized an SVM classifier with:")
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
        self.log.info("Training SVM classifier")
        train_y = np.array(train_y)
        if len(train_y.shape) == 2:
            self.classifier.fit(train_x, np.asarray(train_y).reshape(-1))
        else:
            self.classifier.fit(train_x, train_y)
        self.log.info("Done training SVM classifier")

    def predict(self, test_x):
        """
        Return predicted class labels
        """
        self.log.info("Computing SVM predictions")
        return self.classifier.predict(test_x)

    def predict_proba(self, test_x):
        """
        Return predicted probabilities from SVM classifier
        """

        self.log.info("Computing SVM probabilities")
        return self.classifier.predict_proba(test_x)
