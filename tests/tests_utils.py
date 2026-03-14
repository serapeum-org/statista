import numpy as np
import pytest

from statista.utils import merge_small_bins


class TestMergeSmallBins:

    @pytest.mark.unit
    def test_no_merge_when_all_expected_ge_5(self):
        """
        When all expected bin counts are >= 5, no merging should occur. The function
        should return the same number of bins in the same order, and if the sums of
        observed and expected are already equal, the rescaling step should leave the
        expected counts unchanged.
        """
        observed = [10, 20, 30]
        expected = [12, 18, 30]  # all >= 5 and sums to 60

        merged_obs, merged_exp = merge_small_bins(observed, expected)

        assert isinstance(merged_obs, np.ndarray)
        assert isinstance(merged_exp, np.ndarray)
        np.testing.assert_array_equal(merged_obs, np.array(observed))
        np.testing.assert_array_equal(merged_exp, np.array(expected))
        # sums must match and be unchanged here
        assert merged_obs.sum() == sum(observed) == 60
        assert merged_exp.sum() == merged_obs.sum()

    @pytest.mark.unit
    def test_merge_tail_small_bins_with_previous_large_bin(self):
        """
        Small bins at the right tail (expected < 5) should be accumulated and then
        merged together with the first >= 5 bin encountered when moving left.
        Example: expected = [10, 3, 2] -> all three should merge into one bin.
        """
        observed = [10, 3, 2]
        expected = [10, 3, 2]

        merged_obs, merged_exp = merge_small_bins(observed, expected)

        np.testing.assert_array_equal(merged_obs, np.array([15]))
        np.testing.assert_array_equal(merged_exp, np.array([15]))
        assert merged_exp.sum() == merged_obs.sum() == 15

    @pytest.mark.unit
    def test_leftmost_accumulated_bins_appended_even_if_less_than_5(self):
        """
        If we end the reverse scan with some accumulated small bins and never hit a
        >= 5 bin to flush them into, the function appends that accumulation as a
        single bin (even if its expected count is still < 5).
        Example: expected = [4, 6] -> the leftmost 4 is appended as its own bin.
        """
        observed = [10, 10]
        expected = [4, 6]  # rightmost is >=5; leftmost remains accumulated alone

        merged_obs, merged_exp = merge_small_bins(observed, expected)

        # Order should be low -> high
        np.testing.assert_array_equal(merged_obs, np.array([10, 10]))
        # Before rescaling, expected would be [4, 6]; after rescaling they sum to 20
        np.testing.assert_allclose(
            merged_exp, np.array([8.0, 12.0]), rtol=1e-12, atol=1e-12
        )
        assert merged_exp.sum() == merged_obs.sum() == 20

    @pytest.mark.unit
    def test_accumulate_multiple_small_bins_no_large_bin_then_append(self):
        """
        Multiple consecutive small bins at the far left whose combined expected is
        >= 5 but there is no >= 5 bin to the left should be appended as a single
        merged bin (not combined with any other bin).
        Example: expected = [2, 2, 2, 6] -> becomes [6 (from three 2s), 6].
        """
        observed = [2, 2, 2, 6]
        expected = [2, 2, 2, 6]

        merged_obs, merged_exp = merge_small_bins(observed, expected)

        # No rescaling required (sums are already equal = 12)
        np.testing.assert_array_equal(merged_obs, np.array([6, 6]))
        np.testing.assert_array_equal(merged_exp, np.array([6, 6]))
        assert merged_exp.sum() == merged_obs.sum() == 12

    @pytest.mark.unit
    def test_rescaling_expected_counts_to_observed_total(self):
        """
        Expected counts are rescaled at the end so that their sum matches the total
        observed counts, preserving proportions. This should happen regardless of
        merging.
        """
        observed = [5, 5, 10]
        expected = [2, 3, 5]  # sums to 10, observed sums to 20 -> scale by 2

        merged_obs, merged_exp = merge_small_bins(observed, expected)

        # Here, no merging is needed (all expected >= 2, but rule is <5; small bins exist)
        # Merging logic from the right: [5,5,10] with [2,3,5]
        #   Right bin 5 (>=5) kept; then 3 (<5) accum; then 2 (<5) accum -> append 2+3 as 5
        # Final bins: observed [5, 15], expected before scaling [5, 5] -> after scaling both to sum 20 -> [10, 10]
        np.testing.assert_array_equal(merged_obs, np.array([5 + 5, 10]))
        np.testing.assert_allclose(
            merged_exp, np.array([10.0, 10.0]), rtol=1e-12, atol=1e-12
        )
        assert merged_exp.sum() == merged_obs.sum() == sum(observed)

    @pytest.mark.unit
    def test_output_types_and_ordering(self):
        """
        The function should return numpy arrays and preserve low->high bin ordering
        after performing merges from the right and reversing back.
        """
        observed = [1, 10, 1, 10, 1]
        expected = [1, 10, 1, 10, 1]  # small bins at positions 0,2,4

        merged_obs, merged_exp = merge_small_bins(observed, expected)

        # From right: 1 accum; 10 (>=5) combines -> [11]; then 1 accum; 10 combines -> [11, 11];
        # then leftmost 1 remains -> appended -> [1, 11, 11] after reversing back to low->high
        assert isinstance(merged_obs, np.ndarray)
        assert isinstance(merged_exp, np.ndarray)
        np.testing.assert_array_equal(merged_obs, np.array([1, 11, 11]))
        np.testing.assert_array_equal(merged_exp, np.array([1, 11, 11]))
        # sums preserved and equal between observed and expected
        assert merged_exp.sum() == merged_obs.sum() == sum(observed)
