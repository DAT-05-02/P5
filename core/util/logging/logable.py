from abc import ABC
from core.util.util import aggregate_logger


class Logable(ABC):
    def __init__(self):
        self.log = aggregate_logger(self)

    @classmethod
    def __name__(cls):
        return cls.__module__

    def info(self, msg, **kwargs):
        self.log.info(msg,
                      exc_info=kwargs.get('exc_info', None),
                      stack_info=kwargs.get('stack_info', None))

    def debug(self, msg, **kwargs):
        self.log.debug(msg,
                       exc_info=kwargs.get('exc_info', None),
                       stack_info=kwargs.get('stack_info', None))

    def error(self, msg, **kwargs):
        self.log.error(msg,
                       exc_info=kwargs.get('exc_info', None),
                       stack_info=kwargs.get('stack_info', None))

    def warning(self, msg, **kwargs):
        self.log.warning(msg,
                         exc_info=kwargs.get('exc_info', None),
                         stack_info=kwargs.get('stack_info', None))
