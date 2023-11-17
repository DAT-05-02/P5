import errno
import os
from argparse import Namespace
from functools import wraps
from logging.handlers import TimedRotatingFileHandler
from time import time
import logging
import argparse


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
    logging.getLogger().setLevel(logging.INFO)


def setup_argparse() -> Namespace:
    parser = argparse.ArgumentParser(
        description='MODEL_ID, KERNEL_SIZE, LEARNING_RATE, NUM_EPOCHS, NUM_IMAGES, IMG_SIZE, CROPPED')
    parser.add_argument('MODEL_ID', metavar='ID', type=int,
                        help='an integer for model id')
    parser.add_argument('KERNEL_SIZE', metavar='Kernel', type=int,
                        help='an integer N for kernel size (N, N)')
    parser.add_argument('LEARNING_RATE', metavar='LearningRate', type=float,
                        help='a float for learning rate')
    parser.add_argument('NUM_EPOCHS', metavar='Epochs', type=int,
                        help='an integer for amount of epochs')
    parser.add_argument('NUM_IMAGES', metavar='ImageAmount', type=int,
                        help='an integer for amount of images')
    parser.add_argument('IMG_SIZE', metavar='ImageSize', type=int,
                        help='an integer N for size of images N x N')
    parser.add_argument('CROPPED', metavar='Cropped', type=int,
                        help='crop images with YOLO model (0=false, 1=true)')
    return parser.parse_args()


class LogFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if (
                record.module.startswith('matplotlib') |
                record.module.startswith('urllib3.connectionpool') |
                record.module.startswith('PIL.Image')
        ):
            return False
        return True
