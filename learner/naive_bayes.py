"""An Naive Bayes Implementation"""

from sklearn.naive_bayes import GaussianNB

from .learner import learner

class NaiveBayes(learner):

    """Naive Bayes Wrapper"""

    def __init__(self, alpha):
        super(NaiveBayes, self).__init__()

        self.classifier() = GaussianNB(alpha=0.1)
        self.log.debug("Naive Bayes classifier initialized.")
        self.log.debug('    alpha=%s', alpha)

    def train(self, train_x, train_y):
        """
        Train Naive Bayes classifier
        """
        self.log.info("Training Naive Bayes classifier")

        self.classifier.fit(train_x, train_y)
        self.log.info("Done training Naive Bayes classifier")

    def predict(self, test_x):
        """
        Return predicted class labels
        """
        self.log.info("Computing Naive Bayes predictions")
        return self.classifier.predict(test_x)

    def predict_proba(self, test_x):
        """
        Return predicted probabilities from Naive Bayes classifier
        """

        self.log.info("Computing Naive Bayes probabilities")
        return self.classifier.predict_proba(test_x)