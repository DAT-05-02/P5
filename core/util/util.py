import errno
import os
from functools import wraps
from logging.handlers import TimedRotatingFileHandler
from time import time
import logging


def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print('func:%r args:[%r, %r] took: %2.4f sec' % (f.__name__, args, kw, te - ts))
        return result

    return wrap


def aggregate_logger(self):
    log = logging.getLogger(self.__name__().lower())
    return log


def log_ent_exit(method):
    @wraps(method)
    def _impl(self, *args, **kwargs):
        self.log.debug('Entering: %s', method.__name__)
        tmp = method(self, *args, **kwargs)
        if tmp is not None:
            self.log.debug('%s', tmp)
        self.log.debug('Exiting: %s', method.__name__)
        return tmp

    return _impl


def setup_log(log_level):
    logging.basicConfig(level=log_level,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    try:
        os.makedirs('logs')
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    logger_file = TimedRotatingFileHandler('logs/model.log', when='midnight', interval=1)
    logger_file.setLevel(logging.DEBUG)
    logger_file.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logging.getLogger().addHandler(logger_file)
    logging.getLogger().addFilter(LogFilter())


class LogFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if (
                record.module.startswith('matplotlib')
        ):
            return False
        return True
