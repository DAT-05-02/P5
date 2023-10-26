from abc import ABC
from core.util.util import aggregate_logger


class Logable(ABC):
    def __init__(self):
        self.log = aggregate_logger(self)

    @classmethod
    def __name__(cls):
        return cls.__module__