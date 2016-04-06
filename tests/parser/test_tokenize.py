"""A collection of tests for the tokenizer code"""


def test_tokenize():
    """A smoke test for tokenizer method"""
    from feature_parser import tokenizer

    res = tokenizer.tokenize("This is a test.")
    assert len(res) == 1
    assert res[0] == 'test'
