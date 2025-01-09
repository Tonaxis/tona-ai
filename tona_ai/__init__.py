from tona_ai.core import *
from tona_ai.neat import *

__all__ = ["core", "neat"]

import importlib.metadata

try:
    __version__ = importlib.metadata.version(__package__)
except importlib.metadata.PackageNotFoundError:
    __version__ = ""
