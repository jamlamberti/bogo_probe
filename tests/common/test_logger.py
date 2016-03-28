from common import logger


def test_logger():
    """Test the logger class"""
    import os
    log1 = logger.get_logger("test", "unit_test.log")
    log1.debug("This message is a debug message")
    log1.info("This message is an info message")
    log1.warning("This is a warning message")
    log1.error("This is an error message")
    log1.critical("This is a critical message")
    assert os.path.exists('unit_test.log')
    assert os.path.exists('logs/main.log')
    os.remove('unit_test.log')
