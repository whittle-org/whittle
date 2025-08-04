"""Custom exceptions for Whittle."""

from __future__ import annotations


class IllegalSubNetworkError(ValueError):
    """Exception raised when a sub-network configuration is illegal or invalid.

    This exception is raised when sub-network parameters don't meet the required
    constraints or are incompatible with the super-network configuration.
    """

    pass
