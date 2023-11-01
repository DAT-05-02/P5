import os
import sys

from core.util.logging.logable import Logable


class PySetup(Logable):
    def __init__(self,
                 major_version=3,
                 minor_version=8,
                 micro_version=18,
                 cwd="core"):
        super().__init__()
        self.cwd = cwd
        self.version = (major_version, minor_version, micro_version)
        self._check_py_version()
        self._append_cwd()

    def _check_py_version(self):
        def pretty_version(y): return '.'.join(list(tuple(map(lambda x: str(x), y))))

        if sys.version_info[:3] != self.version:
            raise ImportError(f"Required Python version: {pretty_version(sys.version_info[:3])}\n"
                              f"Current version: {pretty_version(self.version[:3])}")

    def _append_cwd(self):
        sys.path.append(f"{os.getcwd()}{os.sep}{self.cwd}")
        if not os.getcwd().split("/")[-1] == self.cwd:
            os.chdir(self.cwd)
