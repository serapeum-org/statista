from __future__ import annotations

import numpy as np


def merge_small_bins(
    bin_count_observed: list[float], bin_count_fitted_data: list[float]
):
    """Merge small bins for goodness-of-fit tests (e.g., chi-square).

    This utility merges adjacent "small" bins (those whose expected count is < 5)
    starting from the right-most bin and moving left, accumulating small bins
    until their combined expected count is >= 5. If a large (>= 5) bin is
    encountered while there is an accumulation, that accumulation is merged into
    that bin. If the left edge is reached with a remaining accumulation that was
    never merged into a large bin, the accumulation is appended as its own bin.

    After merging, the expected counts are rescaled so that their sum equals the
    total observed count (required by Pearson's chi-square test), preserving the
    expected proportions within the merged structure.

    Args:
        bin_count_observed (List[float]):
            Observed counts per original bin. Must be the same length as
            ``bin_count_fitted_data``. Values should be non-negative.
        bin_count_fitted_data (List[float]):
            Expected (model-fitted) counts per original bin. Must be the same
            length as ``bin_count_observed``. Values should be non-negative.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            Two 1D numpy arrays ``(merged_observed, merged_expected)`` in
            low-to-high bin order after merging and rescaling. The two arrays
            are the same length, and ``merged_expected.sum() ==
            merged_observed.sum()``.

    Raises:
        ZeroDivisionError: If the total expected count across merged bins is 0,
            rescaling cannot be performed (division by zero). This can happen if
            all expected counts are zero.
        ValueError: If the input sequences have different lengths.

    Notes:
        - The function assumes a one-to-one correspondence of observed and
          expected bins. If lengths differ, only a partial zip would occur; to
          avoid silent truncation a ``ValueError`` is raised.
        - Merging proceeds from right to left and the result is then reversed
          back to low-to-high order.
        - The "< 5" rule is a common heuristic for chi-square tests to ensure
          adequate expected counts per bin.

    Examples:
        - Merge tail small bins with the nearest large bin on the left

            ```python
            >>> from statista.utils import merge_small_bins
            >>> merge_small_bins([10, 3, 2], [10, 3, 2])
            (array([15]), array([15.]))

            ```

        - No merging when all expected counts are >= 5

            ```python
            >>> merge_small_bins([10, 20, 30], [12, 18, 30])
            (array([10, 20, 30]), array([12., 18., 30.]))

            ```

        - Accumulated leftmost small bins remain as their own bin if no large bin is found to the left

            ```python
            >>> merge_small_bins([10, 10], [4, 6])
            (array([10, 10]), array([ 8., 12.]))

            ```

        - Expected counts are rescaled to match the observed total while preserving proportions

            ```python
            >>> merge_small_bins([5, 5, 10], [2, 3, 5])
            (array([10, 10]), array([10., 10.]))

            ```
    """
    if len(bin_count_observed) != len(bin_count_fitted_data):
        raise ValueError(
            "bin_count_observed and bin_count_fitted_data must have the same length."
        )

    # Merge tail bins whose expected counts are < 5
    merged_obs: list[float] = []
    merged_exp: list[float] = []
    accum_obs: float = 0
    accum_exp: float = 0

    # Work from the rightmost bin backwards, accumulating bins until the combined
    # expected count is ≥ 5
    for observed, expected in reversed(
        list(zip(bin_count_observed, bin_count_fitted_data))
    ):
        if expected < 5:
            accum_obs += observed
            accum_exp += expected
        else:
            if accum_exp > 0:
                # combine the accumulated small bins with this one
                accum_obs += observed
                accum_exp += expected
                merged_obs.append(accum_obs)
                merged_exp.append(accum_exp)
                accum_obs = accum_exp = 0
            else:
                # keep this bin separate
                merged_obs.append(observed)
                merged_exp.append(expected)

    # Append any remaining accumulated bins
    if accum_exp > 0:
        merged_obs.append(accum_obs)
        merged_exp.append(accum_exp)

    # Reverse the order back to low→high
    arr_obs = np.array(merged_obs[::-1])
    arr_exp = np.array(merged_exp[::-1]).astype(float)

    # Rescale expected counts so they sum to the total number of observations
    # This is required for Pearson’s χ² test
    arr_exp *= arr_obs.sum() / arr_exp.sum()
    return arr_obs, arr_exp
