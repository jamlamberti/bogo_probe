"""A collection of tests for common.config"""

from common import config

# I decided to split the tests up into different
# functions, so we can figure out what went wrong
# faster... pytest doesn't support expect without an
# additional plugin
#   see: http://pythontesting.net/pytest-expect

# I also created a test section in the config file
# these tests should only be based on that section


def test_config_get_str():
    """Check if we can successfully get strings from the config file"""
    test_config = config.Section('testing')
    str_test = test_config.get('s')
    assert str_test == 'abc'
    # Depending on how your environment is configured, it can either
    # be a str or unicode in python2
    assert isinstance(str_test, str) or isinstance(str_test, unicode)


def test_config_get_int():
    """Check if we can successfully get ints from the config file"""
    test_config = config.Section('testing')
    int_test = test_config.getint('x')
    assert isinstance(int_test, int)
    assert int_test == 1


def test_config_get_float():
    """Check if we can successfully get floats from the config file"""
    test_config = config.Section('testing')

    float_test = test_config.getfloat('f')
    assert isinstance(float_test, float)
    assert float_test == -0.05


def test_config_get_bool():
    """Check if we can successfully get booleans from the config file"""
    test_config = config.Section('testing')

    # Lowercase test
    true_test = test_config.getboolean('bt')
    assert true_test
    assert isinstance(true_test, bool)

    # Uppercase test
    false_test = test_config.getboolean('bf')
    assert not false_test
    assert isinstance(false_test, bool)


def test_config_get_list():
    """Check if we can successfully get a list from the config file"""
    test_config = config.Section('testing')
    list_test = test_config.getlist('li')
    assert len(list_test) == 3
    assert isinstance(list_test, list)


def test_config_section():
    """
    Run tests over the testing section of the config file
    """
    pass
