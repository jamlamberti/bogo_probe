"""An SVM implementation"""

import sklearn

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
            gamma='auto',
            coef0=0.0,
        ):
        super(SVM, self).__init__()

        self.classifier = sklearn.svm.SVC(
            kernel=kernel,
            degree=degree,
            gamma=gamma,
            coef0=coef0,
        )
        self.log.debug("Initialized an SVM classifier with:")
        self.log.debug(
            '    kernel=%s, degree=%d, gamma=%s, coef0=%0.3f',
            kernel,
            degree,
            gamma,
            coef0)

    def train(self, train_x, train_y):
        """
        Train the SVM classifier
        """
        self.log.info("Training SVM classifier")

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
