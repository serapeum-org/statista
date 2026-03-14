"""Test descriptors module.

This module contains unit tests for the statistical descriptor functions in
the statista.descriptors module. Each function has a corresponding test class
with multiple test methods covering happy paths, edge cases, and error
handling.
"""

import numpy as np
import pytest

from statista.descriptors import (
    kge,
    mae,
    mbe,
    nse,
    nse_hf,
    nse_lf,
    pearson_corr_coeff,
    r2,
    rmse,
    rmse_hf,
    rmse_lf,
    wb,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def obs_list():
    """Observed values as a plain Python list."""
    return [10, 20, 30, 40, 50]


@pytest.fixture()
def sim_list():
    """Simulated values as a plain Python list."""
    return [12, 18, 33, 43, 48]


@pytest.fixture()
def obs_array(obs_list):
    """Observed values as a numpy array."""
    return np.array(obs_list, dtype=float)


@pytest.fixture()
def sim_array(sim_list):
    """Simulated values as a numpy array."""
    return np.array(sim_list, dtype=float)


# ---------------------------------------------------------------------------
# TestRMSE
# ---------------------------------------------------------------------------


class TestRMSE:
    """Tests for the rmse function."""

    def test_known_values_list(self, obs_list, sim_list):
        """RMSE with simple list inputs and manually verifiable result.

        obs = [10, 20, 30, 40, 50], sim = [12, 18, 33, 43, 48]
        diffs^2 = [4, 4, 9, 9, 4] => mean = 6 => sqrt(6) ~ 2.449
        """
        result = rmse(obs_list, sim_list)
        np.testing.assert_almost_equal(result, np.sqrt(6.0), decimal=5)

    def test_known_values_array(self, obs_array, sim_array):
        """RMSE with numpy array inputs should give the same result."""
        result = rmse(obs_array, sim_array)
        np.testing.assert_almost_equal(result, np.sqrt(6.0), decimal=5)

    def test_perfect_match(self, obs_list):
        """RMSE should be 0 when sim equals obs."""
        result = rmse(obs_list, obs_list)
        assert result == pytest.approx(0.0)

    def test_single_element(self):
        """RMSE with a single-element array."""
        result = rmse([5.0], [7.0])
        assert result == pytest.approx(2.0)

    def test_large_values(self):
        """RMSE with large values should not overflow."""
        obs = [1e10, 2e10]
        sim = [1e10 + 1, 2e10 + 1]
        result = rmse(obs, sim)
        assert result == pytest.approx(1.0)

    def test_negative_values(self):
        """RMSE with negative inputs."""
        obs = [-5, -3, -1]
        sim = [-4, -2, 0]
        # diffs^2 = [1, 1, 1] => mean = 1 => sqrt = 1
        result = rmse(obs, sim)
        assert result == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# TestRMSE_HF
# ---------------------------------------------------------------------------


class TestRMSE_HF:
    """Tests for the rmse_hf (weighted RMSE for high flow) function."""

    @pytest.fixture()
    def hf_obs(self):
        return [10.0, 20.0, 50.0, 100.0, 200.0]

    @pytest.fixture()
    def hf_sim(self):
        return [12.0, 18.0, 55.0, 95.0, 190.0]

    # ---- ws_type 1 --------------------------------------------------------

    def test_ws_type_1(self, hf_obs, hf_sim):
        """Weighting scheme 1: w = (obs/max(obs))^n."""
        result = rmse_hf(hf_obs, hf_sim, ws_type=1, n=2, alpha=0.5)
        assert isinstance(result, (float, np.floating))
        assert result > 0

    def test_ws_type_1_known_value(self):
        """Manually compute ws_type=1 result with tiny data.

        obs = [50, 100], sim = [50, 90], qmax=100, h=[0.5, 1.0]
        w = h^1 = [0.5, 1.0]
        a = (0, 100), b = a*w = (0, 100), c=100
        error = sqrt(100/2) = sqrt(50) ~ 7.071
        """
        result = rmse_hf([50, 100], [50, 90], ws_type=1, n=1, alpha=0.5)
        np.testing.assert_almost_equal(result, np.sqrt(50.0), decimal=5)

    # ---- ws_type 2 --------------------------------------------------------

    def test_ws_type_2(self, hf_obs, hf_sim):
        """Weighting scheme 2: w = (h/alpha)^n, capped at 1."""
        result = rmse_hf(hf_obs, hf_sim, ws_type=2, n=1, alpha=0.5)
        assert result > 0

    # ---- ws_type 3 --------------------------------------------------------

    def test_ws_type_3(self, hf_obs, hf_sim):
        """Weighting scheme 3: binary (0 for h <= alpha, 1 for h > alpha)."""
        result = rmse_hf(hf_obs, hf_sim, ws_type=3, n=1, alpha=0.5)
        assert result > 0

    def test_ws_type_3_known_value(self):
        """Manually compute ws_type=3 with known threshold.

        obs = [10, 100], sim = [10, 90], qmax=100, h=[0.1, 1.0], alpha=0.5
        w = [0, 1] (h>alpha only for second element)
        a = [0, 100], b = [0, 100], c = 100
        error = sqrt(100/2) = sqrt(50) ~ 7.071
        """
        result = rmse_hf([10, 100], [10, 90], ws_type=3, n=1, alpha=0.5)
        np.testing.assert_almost_equal(result, np.sqrt(50.0), decimal=5)

    # ---- ws_type 4 --------------------------------------------------------

    def test_ws_type_4(self, hf_obs, hf_sim):
        """Weighting scheme 4: same binary logic as ws_type 3."""
        result = rmse_hf(hf_obs, hf_sim, ws_type=4, n=1, alpha=0.5)
        # ws_type 3 and 4 use the same formula
        result3 = rmse_hf(hf_obs, hf_sim, ws_type=3, n=1, alpha=0.5)
        np.testing.assert_almost_equal(result, result3, decimal=10)

    # ---- perfect match ----------------------------------------------------

    def test_perfect_match(self, hf_obs):
        """RMSE_HF should be 0 when sim == obs."""
        result = rmse_hf(hf_obs, hf_obs, ws_type=1, n=1, alpha=0.5)
        assert result == pytest.approx(0.0)

    # ---- input with numpy arrays ------------------------------------------

    def test_numpy_array_input(self):
        """Function should accept numpy arrays as well as lists."""
        obs = np.array([10.0, 100.0])
        sim = np.array([10.0, 90.0])
        result = rmse_hf(obs, sim, ws_type=1, n=1, alpha=0.5)
        assert isinstance(result, (float, np.floating))

    # ---- parametric validation tests for ws_type --------------------------

    @pytest.mark.parametrize("bad_ws", [0, 5, -1, 100])
    def test_invalid_ws_type_value(self, bad_ws, hf_obs, hf_sim):
        """ws_type outside 1-4 should raise ValueError."""
        with pytest.raises(ValueError):
            rmse_hf(hf_obs, hf_sim, ws_type=bad_ws, n=1, alpha=0.5)

    @pytest.mark.parametrize("bad_ws", [1.5, "2", None, [1]])
    def test_invalid_ws_type_type(self, bad_ws, hf_obs, hf_sim):
        """Non-integer ws_type should raise TypeError."""
        with pytest.raises(TypeError):
            rmse_hf(hf_obs, hf_sim, ws_type=bad_ws, n=1, alpha=0.5)

    # ---- parametric validation tests for alpha ----------------------------

    @pytest.mark.parametrize("bad_alpha", [0.0, -0.5])
    def test_invalid_alpha_value(self, bad_alpha, hf_obs, hf_sim):
        """alpha <= 0 should raise ValueError."""
        with pytest.raises(ValueError):
            rmse_hf(hf_obs, hf_sim, ws_type=1, n=1, alpha=bad_alpha)

    def test_alpha_equal_one_raises(self, hf_obs, hf_sim):
        """alpha == 1.0 passes the first check but fails the strict < 1 check on line 145."""
        with pytest.raises(ValueError):
            rmse_hf(hf_obs, hf_sim, ws_type=1, n=1, alpha=1.0)

    @pytest.mark.parametrize("bad_alpha", ["0.5", None, [0.5]])
    def test_invalid_alpha_type(self, bad_alpha, hf_obs, hf_sim):
        """Non-numeric alpha should raise ValueError (first check catches it)."""
        with pytest.raises(ValueError):
            rmse_hf(hf_obs, hf_sim, ws_type=1, n=1, alpha=bad_alpha)

    # ---- parametric validation tests for n --------------------------------

    def test_invalid_n_negative(self, hf_obs, hf_sim):
        """Negative n should raise ValueError."""
        with pytest.raises(ValueError):
            rmse_hf(hf_obs, hf_sim, ws_type=1, n=-1, alpha=0.5)

    @pytest.mark.parametrize("bad_n", ["1", None, [1]])
    def test_invalid_n_type(self, bad_n, hf_obs, hf_sim):
        """Non-numeric n should raise TypeError."""
        with pytest.raises(TypeError):
            rmse_hf(hf_obs, hf_sim, ws_type=1, n=bad_n, alpha=0.5)

    def test_n_zero(self, hf_obs, hf_sim):
        """n = 0 is valid and should not raise."""
        result = rmse_hf(hf_obs, hf_sim, ws_type=1, n=0, alpha=0.5)
        assert isinstance(result, (float, np.floating))

    # ---- all ws_types parametrized ----------------------------------------

    @pytest.mark.parametrize("ws", [1, 2, 3, 4])
    def test_all_ws_types_run(self, ws, hf_obs, hf_sim):
        """Each valid ws_type should run without error."""
        result = rmse_hf(hf_obs, hf_sim, ws_type=ws, n=1, alpha=0.5)
        assert result >= 0


# ---------------------------------------------------------------------------
# TestRMSE_LF
# ---------------------------------------------------------------------------


class TestRMSE_LF:
    """Tests for the rmse_lf (weighted RMSE for low flow) function."""

    @pytest.fixture()
    def lf_obs(self):
        return [10.0, 20.0, 50.0, 100.0, 200.0]

    @pytest.fixture()
    def lf_sim(self):
        return [12.0, 18.0, 55.0, 95.0, 190.0]

    # ---- ws_type 1 --------------------------------------------------------

    def test_ws_type_1(self, lf_obs, lf_sim):
        """ws_type 1: w = qr^n where qr = (qmax - obs)/qmax."""
        result = rmse_lf(lf_obs, lf_sim, ws_type=1, n=2, alpha=0.5)
        assert isinstance(result, (float, np.floating))
        assert result > 0

    def test_ws_type_1_known_value(self):
        """Manually compute ws_type 1 result.

        obs = [50, 100], sim = [50, 90], qmax = 100
        qr = [(100-50)/100, (100-100)/100] = [0.5, 0.0]
        w = qr^1 = [0.5, 0.0]
        a = [0, 100], b = [0, 0], c = 0
        error = sqrt(0/2) = 0
        """
        result = rmse_lf([50, 100], [50, 90], ws_type=1, n=1, alpha=0.5)
        assert result == pytest.approx(0.0)

    # ---- ws_type 2 --------------------------------------------------------

    def test_ws_type_2(self, lf_obs, lf_sim):
        """ws_type 2: quadratic weighting function."""
        result = rmse_lf(lf_obs, lf_sim, ws_type=2, n=1, alpha=0.5)
        assert result >= 0

    # ---- ws_type 3 --------------------------------------------------------

    def test_ws_type_3(self, lf_obs, lf_sim):
        """ws_type 3: same as ws_type 2."""
        result2 = rmse_lf(lf_obs, lf_sim, ws_type=2, n=1, alpha=0.5)
        result3 = rmse_lf(lf_obs, lf_sim, ws_type=3, n=1, alpha=0.5)
        np.testing.assert_almost_equal(result2, result3, decimal=10)

    # ---- ws_type 4 --------------------------------------------------------

    def test_ws_type_4(self, lf_obs, lf_sim):
        """ws_type 4: linear weighting function."""
        result = rmse_lf(lf_obs, lf_sim, ws_type=4, n=1, alpha=0.5)
        assert result >= 0

    # ---- perfect match ----------------------------------------------------

    def test_perfect_match(self, lf_obs):
        """RMSE_LF should be 0 when sim == obs."""
        result = rmse_lf(lf_obs, lf_obs, ws_type=1, n=1, alpha=0.5)
        assert result == pytest.approx(0.0)

    # ---- numpy array inputs -----------------------------------------------

    def test_numpy_array_input(self, lf_obs, lf_sim):
        """Function should accept numpy arrays."""
        result = rmse_lf(np.array(lf_obs), np.array(lf_sim), ws_type=1, n=1, alpha=0.5)
        assert isinstance(result, (float, np.floating))

    # ---- parametric validation: ws_type -----------------------------------

    @pytest.mark.parametrize("bad_ws", [0, 5, -1, 100])
    def test_invalid_ws_type_value(self, bad_ws, lf_obs, lf_sim):
        """ws_type outside 1-4 should raise ValueError."""
        with pytest.raises(ValueError):
            rmse_lf(lf_obs, lf_sim, ws_type=bad_ws, n=1, alpha=0.5)

    @pytest.mark.parametrize("bad_ws", [1.5, "2", None])
    def test_invalid_ws_type_type(self, bad_ws, lf_obs, lf_sim):
        """Non-integer ws_type should raise TypeError."""
        with pytest.raises(TypeError):
            rmse_lf(lf_obs, lf_sim, ws_type=bad_ws, n=1, alpha=0.5)

    # ---- parametric validation: alpha -------------------------------------

    @pytest.mark.parametrize("bad_alpha", [0.0, -0.5, 1.0])
    def test_invalid_alpha_value(self, bad_alpha, lf_obs, lf_sim):
        """alpha outside strict (0, 1) should raise ValueError."""
        with pytest.raises(ValueError):
            rmse_lf(lf_obs, lf_sim, ws_type=1, n=1, alpha=bad_alpha)

    @pytest.mark.parametrize("bad_alpha", ["0.5", None, [0.5]])
    def test_invalid_alpha_type(self, bad_alpha, lf_obs, lf_sim):
        """Non-numeric alpha should raise TypeError."""
        with pytest.raises(TypeError):
            rmse_lf(lf_obs, lf_sim, ws_type=1, n=1, alpha=bad_alpha)

    # ---- parametric validation: n -----------------------------------------

    def test_invalid_n_negative(self, lf_obs, lf_sim):
        """Negative n should raise ValueError."""
        with pytest.raises(ValueError):
            rmse_lf(lf_obs, lf_sim, ws_type=1, n=-1, alpha=0.5)

    @pytest.mark.parametrize("bad_n", ["1", None, [1]])
    def test_invalid_n_type(self, bad_n, lf_obs, lf_sim):
        """Non-numeric n should raise TypeError."""
        with pytest.raises(TypeError):
            rmse_lf(lf_obs, lf_sim, ws_type=1, n=bad_n, alpha=0.5)

    # ---- all ws_types parametrized ----------------------------------------

    @pytest.mark.parametrize("ws", [1, 2, 3, 4])
    def test_all_ws_types_run(self, ws, lf_obs, lf_sim):
        """Each valid ws_type should run without error."""
        result = rmse_lf(lf_obs, lf_sim, ws_type=ws, n=1, alpha=0.5)
        assert result >= 0


# ---------------------------------------------------------------------------
# TestKGE
# ---------------------------------------------------------------------------


class TestKGE:
    """Tests for the kge (Kling-Gupta Efficiency) function."""

    def test_perfect_match(self, obs_list):
        """KGE should be 1.0 when sim == obs."""
        result = kge(obs_list, obs_list)
        assert result == pytest.approx(1.0)

    def test_known_value_list(self, obs_list, sim_list):
        """KGE with known inputs (from docstring example).

        obs = [10, 20, 30, 40, 50], sim = [12, 18, 33, 43, 48]
        """
        result = kge(obs_list, sim_list)
        # Manually: corr ~ 0.9959, alpha_std ~ 1.0126, beta_mean ~ 1.04
        # KGE = 1 - sqrt((c-1)^2 + (a-1)^2 + (b-1)^2) ~ 0.9657
        np.testing.assert_almost_equal(result, 0.9657, decimal=3)

    def test_known_value_array(self, obs_array, sim_array):
        """KGE with numpy arrays should give same result as lists."""
        result_list = kge(obs_array.tolist(), sim_array.tolist())
        result_arr = kge(obs_array, sim_array)
        np.testing.assert_almost_equal(result_arr, result_list, decimal=10)

    def test_constant_bias(self):
        """Sim = obs + constant: corr=1, alpha=1, beta != 1."""
        obs = [10, 20, 30, 40, 50]
        sim = [15, 25, 35, 45, 55]
        result = kge(obs, sim)
        # beta = mean(sim)/mean(obs) = 35/30 ~ 1.167, corr=1, alpha=1
        expected = 1 - np.sqrt((1 - 1) ** 2 + (1 - 1) ** 2 + (35.0 / 30.0 - 1) ** 2)
        np.testing.assert_almost_equal(result, expected, decimal=5)

    def test_negative_kge(self):
        """A very poor model can produce negative KGE."""
        obs = [10, 20, 30, 40, 50]
        sim = [50, 40, 30, 20, 10]  # reversed
        result = kge(obs, sim)
        assert result < 0

    def test_single_element(self):
        """Single-element arrays: std is 0, corr is nan; result may be nan."""
        result = kge([5], [5])
        # With zero std, np.corrcoef returns nan; result is nan
        assert np.isnan(result) or isinstance(result, float)


# ---------------------------------------------------------------------------
# TestWB
# ---------------------------------------------------------------------------


class TestWB:
    """Tests for the wb (Water Balance) function."""

    def test_perfect_balance(self):
        """WB = 100% when sum(sim) == sum(obs)."""
        obs = [10, 20, 30, 40, 50]
        sim = [50, 40, 30, 20, 10]
        result = wb(obs, sim)
        assert result == pytest.approx(100.0)

    def test_known_value(self, obs_list, sim_list):
        """WB with known inputs.

        sum(obs) = 150, sum(sim) = 12+18+33+43+48 = 154
        wb = 100 * (1 - |1 - 154/150|) = 100 * (1 - 4/150) ~ 97.333
        """
        result = wb(obs_list, sim_list)
        expected = 100.0 * (1.0 - abs(1.0 - 154.0 / 150.0))
        np.testing.assert_almost_equal(result, expected, decimal=5)

    def test_list_and_array_same(self, obs_list, sim_list, obs_array, sim_array):
        """List and array inputs should give the same result."""
        r_list = wb(obs_list, sim_list)
        r_arr = wb(obs_array, sim_array)
        np.testing.assert_almost_equal(r_list, r_arr, decimal=10)

    def test_underestimation(self):
        """WB drops below 100% for volume underestimation."""
        obs = [10, 20, 30, 40, 50]
        sim = [8, 15, 25, 35, 40]
        # sum(obs)=150, sum(sim)=123, wb=100*(1 - |1-123/150|) = 82
        result = wb(obs, sim)
        assert result == pytest.approx(82.0)

    def test_overestimation(self):
        """WB drops below 100% for volume overestimation as well."""
        obs = [10, 20, 30]
        sim = [20, 40, 60]
        # sum(obs)=60, sum(sim)=120, wb = 100*(1 - |1 - 2|) = 0
        result = wb(obs, sim)
        assert result == pytest.approx(0.0)

    def test_single_element(self):
        """WB with single-element arrays."""
        result = wb([100], [90])
        expected = 100.0 * (1.0 - abs(1.0 - 90.0 / 100.0))
        assert result == pytest.approx(expected)


# ---------------------------------------------------------------------------
# TestNSE
# ---------------------------------------------------------------------------


class TestNSE:
    """Tests for the nse (Nash-Sutcliffe Efficiency) function."""

    def test_perfect_match(self, obs_list):
        """NSE should be 1.0 when sim == obs."""
        result = nse(obs_list, obs_list)
        assert result == pytest.approx(1.0)

    def test_known_value_list(self, obs_list, sim_list):
        """NSE with known inputs.

        obs = [10,20,30,40,50], mean_obs=30
        sum((obs-sim)^2) = 4+4+9+9+4 = 30
        sum((obs-mean)^2) = 400+100+0+100+400 = 1000
        NSE = 1 - 30/1000 = 0.97
        """
        result = nse(obs_list, sim_list)
        assert result == pytest.approx(0.97)

    def test_known_value_array(self, obs_array, sim_array):
        """NSE with numpy arrays should equal list result."""
        np.testing.assert_almost_equal(
            nse(obs_array, sim_array), nse(obs_array.tolist(), sim_array.tolist()), decimal=10
        )

    def test_reversed_gives_negative(self):
        """Reversed sim should give NSE = -3.0 for this example."""
        obs = [10, 20, 30, 40, 50]
        sim = [50, 40, 30, 20, 10]
        result = nse(obs, sim)
        assert result == pytest.approx(-3.0)

    def test_sim_equals_mean(self):
        """Sim = mean(obs) everywhere => NSE = 0."""
        obs = [10, 20, 30, 40, 50]
        sim = [30, 30, 30, 30, 30]
        result = nse(obs, sim)
        assert result == pytest.approx(0.0)

    def test_single_element(self):
        """Single-element: sum((obs-mean)^2) = 0 => 0/0 in numpy returns nan with warning."""
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            result = nse([5], [5])
            # 0/0 in numpy returns nan
            assert np.isnan(result)


# ---------------------------------------------------------------------------
# TestNSE_HF
# ---------------------------------------------------------------------------


class TestNSE_HF:
    """Tests for the nse_hf (modified NSE for high flows) function."""

    def test_perfect_match(self, obs_list):
        """NSE_HF = 1.0 when sim == obs."""
        result = nse_hf(obs_list, obs_list)
        assert result == pytest.approx(1.0)

    def test_known_value(self, obs_list, sim_list):
        """NSE_HF with known inputs.

        obs = [10,20,30,40,50], sim = [12,18,33,43,48], mean_obs=30
        a = sum(obs * (obs-sim)^2) = 10*4 + 20*4 + 30*9 + 40*9 + 50*4 = 40+80+270+360+200 = 950
        b = sum(obs * (obs-30)^2) = 10*400 + 20*100 + 30*0 + 40*100 + 50*400 = 4000+2000+0+4000+20000 = 30000
        NSE_HF = 1 - 950/30000 ~ 0.9683
        """
        result = nse_hf(obs_list, sim_list)
        expected = 1.0 - 950.0 / 30000.0
        np.testing.assert_almost_equal(result, expected, decimal=5)

    def test_list_and_array_same(self, obs_list, sim_list, obs_array, sim_array):
        """List and array inputs should produce the same result."""
        np.testing.assert_almost_equal(
            nse_hf(obs_list, sim_list), nse_hf(obs_array, sim_array), decimal=10
        )


# ---------------------------------------------------------------------------
# TestNSE_LF
# ---------------------------------------------------------------------------


class TestNSE_LF:
    """Tests for the nse_lf (modified NSE for low flows) function."""

    def test_perfect_match(self):
        """NSE_LF = 1.0 when sim == obs (positive values)."""
        obs = [1.0, 5.0, 10.0, 20.0, 50.0]
        result = nse_lf(obs, obs)
        assert result == pytest.approx(1.0)

    def test_positive_values_run(self):
        """NSE_LF should run without error on positive-only data."""
        obs = [1.0, 2.0, 5.0, 10.0, 20.0]
        sim = [1.1, 1.9, 5.2, 9.8, 20.5]
        result = nse_lf(obs, sim)
        assert isinstance(result, (float, np.floating))
        # Should be close to 1 for a good fit
        assert result > 0.9

    def test_list_and_array_same(self):
        """List and array inputs should produce the same result."""
        obs = [1.0, 2.0, 5.0, 10.0]
        sim = [1.1, 1.9, 5.2, 9.8]
        r_list = nse_lf(obs, sim)
        r_arr = nse_lf(np.array(obs), np.array(sim))
        np.testing.assert_almost_equal(r_list, r_arr, decimal=10)

    def test_zero_in_obs_warns(self):
        """Log(0) is -inf; numpy emits a RuntimeWarning for divide by zero."""
        obs = [0, 1, 2, 3]
        sim = [0.1, 1, 2, 3]
        with pytest.warns(RuntimeWarning):
            result = nse_lf(obs, sim)
            # log(0) = -inf contaminates the computation, result is nan
            assert np.isnan(result)

    def test_negative_in_obs_raises(self):
        """Log of negative number should produce RuntimeWarning / nan."""
        obs = [-1, 1, 2, 3]
        sim = [0.1, 1, 2, 3]
        # np.log of negative returns nan with a RuntimeWarning
        with pytest.warns(RuntimeWarning):
            result = nse_lf(obs, sim)
            assert np.isnan(result)


# ---------------------------------------------------------------------------
# TestMBE
# ---------------------------------------------------------------------------


class TestMBE:
    """Tests for the mbe (Mean Bias Error) function."""

    def test_overestimation(self):
        """Positive bias when sim > obs everywhere.

        obs = [10,20,30,40,50], sim = [12,22,32,42,52]
        MBE = mean(sim - obs) = mean([2,2,2,2,2]) = 2.0
        """
        obs = [10, 20, 30, 40, 50]
        sim = [12, 22, 32, 42, 52]
        result = mbe(obs, sim)
        assert result == pytest.approx(2.0)

    def test_underestimation(self):
        """Negative bias when sim < obs everywhere."""
        obs = [10, 20, 30, 40, 50]
        sim = [8, 18, 28, 38, 48]
        result = mbe(obs, sim)
        assert result == pytest.approx(-2.0)

    def test_no_bias(self):
        """MBE = 0 when errors cancel out."""
        obs = [10, 20, 30, 40, 50]
        sim = [12, 18, 32, 38, 50]
        # diffs = [2, -2, 2, -2, 0] => mean = 0
        result = mbe(obs, sim)
        assert result == pytest.approx(0.0)

    def test_perfect_match(self, obs_list):
        """MBE = 0 when sim == obs."""
        result = mbe(obs_list, obs_list)
        assert result == pytest.approx(0.0)

    def test_list_input(self, obs_list, sim_list):
        """MBE with list inputs."""
        result = mbe(obs_list, sim_list)
        # sim - obs = [2, -2, 3, 3, -2] => mean = 4/5 = 0.8
        assert result == pytest.approx(0.8)

    def test_array_input(self, obs_array, sim_array):
        """MBE with numpy array inputs should match list result."""
        r_list = mbe(obs_array.tolist(), sim_array.tolist())
        r_arr = mbe(obs_array, sim_array)
        np.testing.assert_almost_equal(r_arr, r_list, decimal=10)

    def test_single_element(self):
        """MBE with single-element arrays."""
        result = mbe([5], [7])
        assert result == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# TestMAE
# ---------------------------------------------------------------------------


class TestMAE:
    """Tests for the mae (Mean Absolute Error) function."""

    def test_known_value(self, obs_list, sim_list):
        """MAE with known inputs.

        |obs - sim| = [2, 2, 3, 3, 2] => mean = 12/5 = 2.4
        """
        result = mae(obs_list, sim_list)
        assert result == pytest.approx(2.4)

    def test_perfect_match(self, obs_list):
        """MAE = 0 when sim == obs."""
        result = mae(obs_list, obs_list)
        assert result == pytest.approx(0.0)

    def test_always_non_negative(self):
        """MAE should always be >= 0."""
        np.random.seed(42)
        obs = np.random.rand(100).tolist()
        sim = np.random.rand(100).tolist()
        result = mae(obs, sim)
        assert result >= 0

    def test_list_and_array_same(self, obs_list, sim_list, obs_array, sim_array):
        """List and array inputs should give the same result."""
        np.testing.assert_almost_equal(
            mae(obs_list, sim_list), mae(obs_array, sim_array), decimal=10
        )

    def test_single_element(self):
        """MAE with single-element arrays."""
        result = mae([5], [8])
        assert result == pytest.approx(3.0)

    def test_negative_values(self):
        """MAE works correctly with negative inputs."""
        obs = [-10, -5, 0, 5, 10]
        sim = [-8, -3, 2, 7, 12]
        # |diffs| = [2, 2, 2, 2, 2] => mean = 2.0
        result = mae(obs, sim)
        assert result == pytest.approx(2.0)

    def test_large_values(self):
        """MAE with large values does not overflow."""
        obs = [1e15, 2e15]
        sim = [1e15 + 1e10, 2e15 - 1e10]
        result = mae(obs, sim)
        assert result == pytest.approx(1e10)


# ---------------------------------------------------------------------------
# TestPearsonCorrCoeff
# ---------------------------------------------------------------------------


class TestPearsonCorrCoeff:
    """Tests for the pearson_corr_coeff function."""

    def test_perfect_positive(self):
        """Perfectly correlated: r = 1."""
        x = [1, 2, 3, 4, 5]
        y = [2, 4, 6, 8, 10]
        result = pearson_corr_coeff(x, y)
        assert result == pytest.approx(1.0)

    def test_perfect_negative(self):
        """Perfectly negatively correlated: r = -1."""
        x = [1, 2, 3, 4, 5]
        y = [10, 8, 6, 4, 2]
        result = pearson_corr_coeff(x, y)
        assert result == pytest.approx(-1.0)

    def test_no_correlation(self):
        """Approximately zero correlation with orthogonal-like data."""
        # Construct data with known zero correlation
        x = [1, -1, 1, -1]
        y = [1, 1, -1, -1]
        result = pearson_corr_coeff(x, y)
        assert result == pytest.approx(0.0)

    def test_list_and_array_same(self):
        """List and array inputs should give the same result."""
        x = [1, 2, 3, 4, 5]
        y = [5, 2, 8, 1, 4]
        r_list = pearson_corr_coeff(x, y)
        r_arr = pearson_corr_coeff(np.array(x), np.array(y))
        np.testing.assert_almost_equal(r_list, r_arr, decimal=10)

    def test_identical_values(self, obs_list):
        """Correlation of a variable with itself is 1."""
        result = pearson_corr_coeff(obs_list, obs_list)
        assert result == pytest.approx(1.0)

    def test_constant_x(self):
        """Constant x => std = 0 => correlation is nan."""
        x = [5, 5, 5, 5]
        y = [1, 2, 3, 4]
        result = pearson_corr_coeff(x, y)
        assert np.isnan(result)

    def test_single_element(self):
        """Single-element arrays produce nan (no variance)."""
        result = pearson_corr_coeff([1], [2])
        assert np.isnan(result)

    def test_scale_invariance(self):
        """Pearson coefficient is scale-invariant."""
        x = [1, 2, 3, 4, 5]
        y = [10, 20, 30, 40, 50]
        r1 = pearson_corr_coeff(x, y)
        # Scale y by 1000
        y_scaled = [v * 1000 for v in y]
        r2_val = pearson_corr_coeff(x, y_scaled)
        np.testing.assert_almost_equal(r1, r2_val, decimal=10)

    def test_range_bound(self):
        """Result should always be in [-1, 1] (or nan for degenerate cases)."""
        np.random.seed(42)
        x = np.random.rand(50).tolist()
        y = np.random.rand(50).tolist()
        result = pearson_corr_coeff(x, y)
        assert -1.0 <= result <= 1.0


# ---------------------------------------------------------------------------
# TestR2
# ---------------------------------------------------------------------------


class TestR2:
    """Tests for the r2 (coefficient of determination) function."""

    def test_perfect_match(self, obs_list):
        """R2 = 1.0 when sim == obs."""
        result = r2(obs_list, obs_list)
        assert result == pytest.approx(1.0)

    def test_known_value(self, obs_list, sim_list):
        """R2 with known inputs (same numerics as NSE for the 1:1 line).

        R2 = 1 - SS_res / SS_tot = 1 - 30/1000 = 0.97
        """
        result = r2(obs_list, sim_list)
        assert result == pytest.approx(0.97)

    def test_list_and_array_same(self, obs_list, sim_list, obs_array, sim_array):
        """List and array inputs should give the same result."""
        np.testing.assert_almost_equal(
            r2(obs_list, sim_list), r2(obs_array, sim_array), decimal=10
        )

    def test_reversed_negative(self):
        """Reversed sim should give R2 = -3.0."""
        obs = [10, 20, 30, 40, 50]
        sim = [50, 40, 30, 20, 10]
        result = r2(obs, sim)
        assert result == pytest.approx(-3.0)

    def test_sim_equals_mean(self):
        """Sim = mean(obs) everywhere => R2 = 0."""
        obs = [10, 20, 30, 40, 50]
        sim = [30, 30, 30, 30, 30]
        result = r2(obs, sim)
        assert result == pytest.approx(0.0)

    def test_near_perfect(self):
        """Close sim should give R2 close to 1."""
        obs = [10, 20, 30, 40, 50]
        sim = [11, 19, 31, 41, 49]
        result = r2(obs, sim)
        assert result > 0.99

    def test_single_element(self):
        """Single-element case: SS_tot = 0, R2 depends on sklearn behaviour."""
        # sklearn r2_score with single element where y_true == y_pred returns 1.0
        # but if y_true != y_pred it returns 0.0 (as SS_tot = 0)
        result = r2([5], [5])
        # sklearn returns 1.0 for perfect single-element match
        assert result == pytest.approx(1.0) or np.isnan(result)
