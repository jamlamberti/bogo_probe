"""A Knn Implementation"""

from sklearn.neighbors import KNeighborsClassifier

from .learner import Learner


class KNN(Learner):

    """Knn Wrapper"""

    def __init__(self, k=5):
        super(KNN, self).__init__()

        self.classifier = KNeighborsClassifier(n_neighbors=k)
        self.log.debug("Knn classifier initialized.")

    def train(self, train_x, train_y):
        """
        Train Knn classifier
        """
        self.log.info("Training Knn classifier")

        self.classifier.fit(train_x, train_y)
        self.log.info("Done training Knn classifier")

    def predict(self, test_x):
        """
        Return predicted class labels
        """
        self.log.info("Computing Knn predictions")
        return self.classifier.predict(test_x)

    def predict_proba(self, test_x):
        """
        Return predicted probabilities from Knn classifier
        """

        self.log.info("Computing Knn probabilities")
        return self.classifier.predict_proba(test_x)
