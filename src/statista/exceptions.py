"""Custom exceptions for the statista package."""


class StatistaError(Exception):
    """Base exception for all statista errors."""


class ParameterError(StatistaError, ValueError):
    """Invalid or missing distribution parameters.

    Raised when:
    - scale is not positive
    - a dict has invalid keys for Parameters conversion
    - parameters have not been estimated before a GoF test
    """
