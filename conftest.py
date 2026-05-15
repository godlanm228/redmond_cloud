import ctypes.util
import sys

# на Windows find_library("c") часто возвращает None → CDLL(None) падает
# заставляем его возвращать msvcrt.dll
if sys.platform.startswith("win"):
    if ctypes.util.find_library("c") is None:
        _orig_find_library = ctypes.util.find_library
        ctypes.util.find_library = lambda name: "msvcrt" if name == "c" else _orig_find_library(name)
