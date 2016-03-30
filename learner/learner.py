"""Generic Learner Class"""

from os.path import join, abspath

from common import logger, config


class Learner(object):

    """
    A wrapper around all ML classes
    """

    def __init__(self):
        """
        Initialize the classifier with whatever hyperparams you want
        """

        # I am currently throwing these logs into the log-dir
        # (specified in the config file)
        # It would probably be better to have test specific logging
        # also placed into the results dir

        logging_config = config.Section('logging')
        self.log = logger.get_logger(
            'learner',
            join(
                abspath(logging_config.get('log-dir')),
                'learner.log'))

    def train(self, train_x, train_y):
        """
        Train the classifier
        """
        raise NotImplementedError("Train method not implemented")

    def predict(self, test_x):
        """
        Return predicted class labels
        """
        raise NotImplementedError("Predict method not implemented")

    def predict_proba(self, test_x):
        """
        Return predicted probabilities
        """
        raise NotImplementedError("Predict Probabilities not implemented")
