"""statista - statistics package."""

from importlib.metadata import PackageNotFoundError, version

from statista.exceptions import ParameterError, StatistaError

try:
    __version__ = version(__name__)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"
