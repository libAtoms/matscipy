import os
import logging


def create_logger(name, level='warning', log_file=None):
    """
    Create a logger with defined level, and some nice formatter and handler.

    Parameter:
    ----------
    name: str
        Name of the logger.

    level: str, optional default='warning'
        The sensitivity level of both the logger and the handler.
        Allowed options are 'debug', 'info', 'warning', 'error' and 'critical'.

    log_file: str, optional default=None
        Path to the file, in case you want to log to a file rather than standard out.

    Returns:
    --------
    logger: logging.Logger
        The respective logger.
    """

    levels = {'debug':      logging.DEBUG,
              'info':       logging.INFO,
              'warning':    logging.WARNING,
              'error':      logging.ERROR,
              'critical':   logging.CRITICAL}

    logger = logging.getLogger(name)
    logger.setLevel(levels[level])

    # create console handler and set level to debug
    if not log_file:
        ch = logging.StreamHandler()
    else:
        ch = logging.FileHandler(os.path.abspath(log_file))

    ch.setLevel(levels[level])

    # create formatter
    formatter = logging.Formatter('%(asctime)s: [%(levelname)s] %(name)s : %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger


def set_logging(level='warning', log_file=None):
    """
    Set the global logging level (and corresponding file location).

    Parameter:
    ----------
    level: str, optional default='warning'
        The sensitivity level of both the logger and the handler.
        Allowed options are 'debug', 'info', 'warning', 'error' and 'critical'.

    log_file: str, optional default=None
        Path to the file, in case you want to log to a file rather than standard out.
    """
    logger = create_logger('matscipy.calculators.committee', level=level, log_file=log_file)
