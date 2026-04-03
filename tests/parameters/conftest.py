"""Shared fixtures for Lmoments parameter estimation tests."""

import pytest


@pytest.fixture(scope="module")
def sample_data():
    """Sample data for testing Lmoments calculation."""
    return [1.2, 3.4, 5.6, 7.8, 9.0, 2.3, 4.5, 6.7, 8.9]


@pytest.fixture(scope="module")
def expected_lmoments():
    """Expected L-moments for sample_data."""
    return [
        5.488888888888889,
        1.722222222222222,
        -0.06451612903225806,
        -0.0645161290322581,
        -0.0645161290322581,
    ]
