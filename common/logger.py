"""Log Handler Implementation"""
import os
import logging

from . import config, util


def get_logger(
        log_type,
        log_file,
        stream_level=logging.ERROR,
        file_level=logging.DEBUG):
    """
    Return a logger
        - log_type: name to display
        - log_file: file to log to
        - stream_level: what level of logging should be displayed to the screen
        - file_level: what level of logging should be sent to the file
    """
    logging_config = config.Section('logging')
    logger = logging.getLogger(log_type)
    logger.setLevel(logging.DEBUG)

    direc = os.path.abspath(logging_config.get('log-dir'))
    util.make_dir(direc)

    main_handler = logging.FileHandler(
        os.path.join(direc, logging_config.get('main-log')))
    main_level = logging_config.get('main-level')

    if not hasattr(logging, main_level):
        raise Exception('%s is not a valid log level!' % main_level)

    main_handler.setLevel(getattr(logging, main_level))

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(file_level)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(stream_level)

    # create formatter and add it to the handlers
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)8s - %(name)s - '
        '%(filename)s:%(lineno)s - %(funcName)s - %(message)s')
    main_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    # add the handlers to the logger
    logger.addHandler(main_handler)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger
