def test_config_section():
    """Run tests over the testing section of the config file"""
    from common import config

    test_config = config.Section('testing')

    str_test = test_config.get('s')
    assert str_test == 'abc'
    assert isinstance(str_test, str) or isinstance(str_test, unicode)

    int_test = test_config.getint('x')
    assert isinstance(int_test, int)
    assert int_test == 1

    float_test = test_config.getfloat('f')
    assert isinstance(float_test, float)
    assert float_test == -0.05

    true_test = test_config.getboolean('bt')
    assert true_test
    assert isinstance(true_test, bool)

    false_test = test_config.getboolean('bf')
    assert not false_test
    assert isinstance(false_test, bool)

    list_test = test_config.getlist('li')
    assert len(list_test) == 3
    assert isinstance(list_test, list)
