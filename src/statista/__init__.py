try:
    from importlib.metadata import PackageNotFoundError, version  # type: ignore
except ImportError:  # pragma: no cover
    from importlib_metadata import (  # type: ignore[no-redef,assignment]
        PackageNotFoundError,
        version,
    )


try:
    __version__ = version(__name__)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"

# documentation format
__author__ = "Mostafa Farrag"
__email__ = "moah.farag@gmail.com"
__docformat__ = "restructuredtext"

__metadata__ = {
    "author": __author__,
    "email": __email__,
    "version": __version__,
    "docformat": __docformat__,
}


__doc__ = """
statista - statistics package
"""
