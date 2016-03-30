"""A collection of tests for learner.learner"""
import pytest

from learner import learner


def test_learner():
    """
    Check that learner doesn't do anything useful
        After all, it is just a template
    """

    learn = learner.Learner()

    with pytest.raises(NotImplementedError):
        learn.train(None, None)

    with pytest.raises(NotImplementedError):
        learn.predict(None)

    with pytest.raises(NotImplementedError):
        learn.predict_proba(None)
