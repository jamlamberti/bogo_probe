"""Generic Learner Class"""

class Learner(object):
    """
    A wrapper around all ML classes
    """

    def __init__(self):
        """
        Initialize the classifier with whatever hyperparams you want
        """
        pass

    def train(self, train_x, train_y):
        """
        Train the classifier
        """
        raise NotImplementedError("Train method not implemented")

    def predict(x):
        """
        Return predicted class labels
        """
        raise NotImplementedError("Predict method not implemented")

    def predict_proba(x):
        """
        Return predicted probabilities
        """
        raise NotImplementedError("Predict Probabilities not implemented")
