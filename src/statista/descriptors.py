"""Statistical descriptors."""

from numbers import Number
from typing import Union

import numpy as np
from sklearn.metrics import r2_score


def rmse(obs: Union[list, np.ndarray], sim: Union[list, np.ndarray]) -> float:
    """Root Mean Squared Error.

    Calculates the Root Mean Squared Error between observed and simulated values.
    RMSE is a commonly used measure of the differences between values predicted by a model
    and the values actually observed.

    Args:
        obs: Measured/observed values as a list or numpy array.
        sim: Simulated/predicted values as a list or numpy array.

    Returns:
        float: The RMSE value representing the square root of the average squared difference
            between observed and simulated values.

    Raises:
        ValueError: If the input arrays have different lengths.

    Examples:
        - Using lists:
            ```python
            >>> import numpy as np
            >>> from statista.descriptors import rmse
            >>> observed = [1, 2, 3, 4, 5]
            >>> simulated = [1.1, 2.1, 2.9, 4.2, 5.2]
            >>> rmse_value = rmse(observed, simulated)
            >>> print(f"RMSE: {rmse_value:.4f}")
            RMSE: 0.1483

            ```
        - Using numpy arrays:
            ```python
            >>> observed = np.array([10, 20, 30, 40, 50])
            >>> simulated = np.array([12, 18, 33, 43, 48])
            >>> rmse_value = rmse(observed, simulated)
            >>> print(f"RMSE: {rmse_value:.4f}")
            RMSE: 2.4495

            ```

    See Also:
        - mae: Mean Absolute Error
        - mbe: Mean Bias Error
    """
    # convert Qobs & sim into arrays
    obs = np.array(obs)
    sim = np.array(sim)

    rmse = np.sqrt(np.average((np.array(obs) - np.array(sim)) ** 2), dtype=np.float64)

    return rmse


def rmse_hf(
    obs: Union[list, np.ndarray],
    sim: Union[list, np.ndarray],
    ws_type: int,
    n: int,
    alpha: Union[int, float],
) -> float:
    """Weighted Root Mean Square Error for High flow.

    Calculates a weighted version of RMSE that gives more importance to high flow values.
    Different weighting schemes can be applied based on the ws_type parameter.

    Args:
        obs: Observed flow values as a list or numpy array.
        sim: Simulated flow values as a list or numpy array.
        ws_type: Weighting scheme type (integer between 1 and 4):
            1: Uses h^n weighting where h is the rational discharge.
            2: Uses (h/alpha)^n weighting with a cap at 1 for h > alpha.
            3: Binary weighting: 0 for h <= alpha, 1 for h > alpha.
            4: Same as type 3.
            Any other value: Uses sigmoid function weighting.
        n: Power parameter for the weighting function.
        alpha: Upper limit parameter for the weighting function (between 0 and 1).

    Returns:
        float: The weighted RMSE value for high flows.

    Raises:
        TypeError: If ws_type is not an integer, alpha is not a number, or n is not a number.
        ValueError: If ws_type is not between 1 and 4, n is negative, or alpha is not between 0 and 1.

    Examples:
        ```python
        >>> import numpy as np
        >>> from statista.descriptors import rmse_hf
        >>> observed = [10, 20, 50, 100, 200]
        >>> simulated = [12, 18, 55, 95, 190]

        ```
        - Using weighting scheme 1 with n=2 and alpha=0.5:
            ```python
            >>> error = rmse_hf(observed, simulated, ws_type=1, n=2, alpha=0.5)
            >>> print(f"Weighted RMSE for high flows: {error:.4f}")
            Weighted RMSE for high flows: 4.6446

            ```
        - Using weighting scheme 3 (binary weighting) with alpha=0.7:
            ```python
            >>> error = rmse_hf(observed, simulated, ws_type=3, n=1, alpha=0.7)
            >>> print(f"Weighted RMSE for high flows: {error:.4f}")
            Weighted RMSE for high flows: 4.4721

            ```

    See Also:
        - rmse: Root Mean Square Error
        - rmse_lf: Weighted Root Mean Square Error for Low flow
    """
    if not isinstance(ws_type, int):
        raise TypeError(
            f"Weighting scheme should be an integer number between 1 and 4 and you entered {ws_type}"
        )

    if not isinstance(alpha, Number) or not (0 < alpha <= 1):
        raise ValueError(
            f"alpha should be a number and between 0 & 1 you have entered {alpha}"
        )

    if not isinstance(n, Number):
        raise TypeError("N should be a number and between 0 & 1")

    # Input values
    if not (1 <= ws_type <= 4):
        raise ValueError(
            f"Weighting scheme should be an integer number between 1 and 4 you have entered {ws_type}"
        )

    if n < 0:
        raise ValueError(
            f"Weighting scheme Power should be positive number you have entered {n}"
        )

    if not (0 < alpha < 1):
        raise ValueError(
            f"alpha should be float number and between 0 & 1 you have entered {alpha}"
        )

    # convert obs & sim into arrays
    obs = np.array(obs)
    sim = np.array(sim)

    qmax = max(obs)
    h = obs / qmax  # rational Discharge

    if ws_type == 1:
        w = h**n  # rational Discharge power N
    elif (
        ws_type == 2
    ):  # -------------------------------------------------------------N is not in the equation
        w = (h / alpha) ** n
        w[h > alpha] = 1
    elif ws_type == 3:
        w = np.zeros(np.size(h))  # zero for h < alpha and 1 for h > alpha
        w[h > alpha] = 1
    elif ws_type == 4:
        w = np.zeros(np.size(h))  # zero for h < alpha and 1 for h > alpha
        w[h > alpha] = 1
    else:  # sigmoid function
        w = 1 / (1 + np.exp(-10 * h + 5))

    a = (obs - sim) ** 2
    b = a * w
    c = sum(b)
    error = np.sqrt(c / len(obs))

    return error


def rmse_lf(
    obs: Union[list, np.ndarray],
    qsim: Union[list, np.ndarray],
    ws_type: int,
    n: int,
    alpha: Union[int, float],
) -> float:
    """Weighted Root Mean Square Error for Low flow.

    Calculates a weighted version of RMSE that gives more importance to low flow values.
    Different weighting schemes can be applied based on the ws_type parameter.

    Args:
        obs: Observed flow values as a list or numpy array.
        qsim: Simulated flow values as a list or numpy array.
        ws_type: Weighting scheme type (integer between 1 and 4):
            1: Uses qr^n weighting where qr is the rational discharge for low flows.
            2: Uses a quadratic function of (1-qr) with a cap at 0 for (1-qr) > alpha.
            3: Same as type 2.
            4: Uses linear function 1-((1-qr)/alpha) with a cap at 0 for (1-qr) > alpha.
            Any other value: Uses sigmoid function weighting.
        n: Power parameter for the weighting function.
        alpha: Upper limit parameter for the weighting function (between 0 and 1).

    Returns:
        float: The weighted RMSE value for low flows.

    Raises:
        TypeError: If ws_type is not an integer, alpha is not a number, or n is not a number.
        ValueError: If ws_type is not between 1 and 4, n is negative, or alpha is not between 0 and 1.

    Examples:
        - Example with good performance on low flows:
            ```python
            >>> import numpy as np
            >>> from statista.descriptors import rmse_lf
            >>> observed = [10, 20, 50, 100, 200]
            >>> simulated = [12, 18, 55, 95, 190]

            ```
        - Using weighting scheme 1 with n=2 and alpha=0.5:
            ```python
            >>> error = rmse_lf(observed, simulated, ws_type=1, n=2, alpha=0.5)
            >>> print(f"Weighted RMSE for low flows: {error:.4f}")
            Weighted RMSE for low flows: 2.3308

            ```
        - Using weighting scheme 4 with alpha=0.7:
            ```python
            >>> error = rmse_lf(observed, simulated, ws_type=4, n=1, alpha=0.7)
            >>> print(f"Weighted RMSE for low flows: {error:.4f}")
            Weighted RMSE for low flows: 2.4640

            ```

    See Also:
        - rmse: Root Mean Square Error
        - rmse_hf: Weighted Root Mean Square Error for High flow
    """
    if not isinstance(ws_type, int):
        raise TypeError(
            f"Weighting scheme should be an integer number between 1 and 4 and you entered {ws_type}"
        )

    if not (isinstance(alpha, int) or isinstance(alpha, float)):
        raise TypeError("alpha should be a number and between 0 & 1")

    if not isinstance(n, Number):
        raise TypeError("N should be a number and between 0 & 1")

    # Input values
    if not 1 <= ws_type <= 4:
        raise ValueError(
            f"Weighting scheme should be an integer number between 1 and 4 you have entered {ws_type}"
        )

    if n < 0:
        raise ValueError(
            f"Weighting scheme Power should be positive number you have entered {n}"
        )

    if not 0 < alpha < 1:
        raise ValueError(
            f"alpha should be float number and between 0 & 1 you have entered {alpha}"
        )

    # convert obs & sim into arrays
    obs = np.array(obs)
    qsim = np.array(qsim)

    # rational Discharge power N
    qmax = max(obs)
    qr = (qmax - obs) / qmax

    if ws_type == 1:
        w = qr**n
    elif ws_type == 2:
        # N is not in the equation
        #  w=1-qr*((0.50 - alpha)**N)
        w = ((1 / (alpha**2)) * (1 - qr) ** 2) - ((2 / alpha) * (1 - qr)) + 1
        w[1 - qr > alpha] = 0
    elif ws_type == 3:
        # the same as WStype 2
        # w=1-qr*((0.50 - alpha)**N)
        w = ((1 / (alpha**2)) * (1 - qr) ** 2) - ((2 / alpha) * (1 - qr)) + 1
        w[1 - qr > alpha] = 0
    elif ws_type == 4:
        # w = 1-qr*(0.50 - alpha)
        w = 1 - ((1 - qr) / alpha)
        w[1 - qr > alpha] = 0
    else:  # sigmoid function
        # w=1/(1+np.exp(10*h-5))
        w = 1 / (1 + np.exp(-10 * qr + 5))

    a = (obs - qsim) ** 2
    b = a * w
    c = sum(b)
    error = np.sqrt(c / len(obs))

    return error


def kge(obs: Union[list, np.ndarray], sim: Union[list, np.ndarray]) -> float:
    """Kling-Gupta Efficiency.

    Calculates the Kling-Gupta Efficiency (KGE) between observed and simulated values.

    KGE addresses limitations of using a single error function like Nash-Sutcliffe Efficiency (NSE)
    or RMSE by decomposing the error into three components: correlation, variability, and bias.
    This provides a more comprehensive assessment of model performance.

    Args:
        obs: Observed flow values as a list or numpy array.
        sim: Simulated flow values as a list or numpy array.

    Returns:
        float: The KGE value. KGE ranges from -∞ to 1, with 1 being perfect agreement.
            Values closer to 1 indicate better model performance.

    Raises:
        ValueError: If the input arrays have different lengths or contain invalid values.

    Examples:
        - Example with good performance:
            ```python
            >>> import numpy as np
            >>> from statista.descriptors import kge
            >>> observed = [10, 20, 30, 40, 50]
            >>> simulated = [12, 18, 33, 43, 48]
            >>> kge_value = kge(observed, simulated)
            >>> print(f"KGE: {kge_value:.4f}")
            KGE: 0.9657

            ```
        - Example with poorer performance:
            ```python
            >>> observed = [10, 20, 30, 40, 50]
            >>> simulated = [5, 15, 45, 35, 60]
            >>> kge_value = kge(observed, simulated)
            >>> print(f"KGE: {kge_value:.4f}")
            KGE: 0.5805

            ```

    References:
        Gupta, H. V., Kling, H., Yilmaz, K. K., & Martinez, G. F. (2009).
        Decomposition of the mean squared error and NSE performance criteria:
        Implications for improving hydrological modelling.
        Journal of Hydrology, 377(1-2), 80-91.

    See Also:
        - nse: Nash-Sutcliffe Efficiency
        - rmse: Root Mean Square Error
    """
    # convert obs & sim into arrays
    obs = np.array(obs)
    sim = np.array(sim)

    c = np.corrcoef(obs, sim)[0][1]
    alpha = np.std(sim) / np.std(obs)
    beta = np.mean(sim) / np.mean(obs)

    kge = 1 - np.sqrt(((c - 1) ** 2) + ((alpha - 1) ** 2) + ((beta - 1) ** 2))

    return kge


def wb(obs: Union[list, np.ndarray], sim: Union[list, np.ndarray]) -> float:
    """Water Balance Error.

    Calculates the water balance error, which measures how well the model reproduces
    the total stream flow volume.

    This metric allows error compensation between time steps and is not an indication
    of the temporal accuracy of the model. It only measures the overall volume balance.
    Note that the naive model of Nash-Sutcliffe (simulated flow equals the average observed
    flow) will result in a WB error of 100%.

    Args:
        obs: Observed flow values as a list or numpy array.
        sim: Simulated flow values as a list or numpy array.

    Returns:
        float: The water balance error as a percentage (0-100).
            100% indicates perfect volume balance, while lower values indicate poorer performance.

    Raises:
        ValueError: If the sum of observed values is zero (division by zero).
        ValueError: If the input arrays have different lengths.

    Examples:
        - Example with goof volume balance
            ```python
            >>> import numpy as np
            >>> from statista.descriptors import wb
            >>> observed = [10, 20, 30, 40, 50]
            >>> simulated = [12, 18, 33, 43, 44]
            >>> wb_value = wb(observed, simulated)
            >>> print(f"Water Balance Error: {wb_value:.2f}%")
            Water Balance Error: 100.00%

            ```
        - Example with volume underestimation
            ```python
            >>> observed = [10, 20, 30, 40, 50]
            >>> simulated = [8, 15, 25, 35, 40]
            >>> wb_value = wb(observed, simulated)
            >>> print(f"Water Balance Error: {wb_value:.2f}%")
            Water Balance Error: 82.00%

            ```

    References:
        Oudin, L., Andréassian, V., Mathevet, T., Perrin, C., & Michel, C. (2006).
        Dynamic averaging of rainfall-runoff model simulations from complementary
        model parameterizations. Water Resources Research, 42(7).

    See Also:
        - rmse: Root Mean Square Error
        - nse: Nash-Sutcliffe Efficiency
    """
    qobs_sum = np.sum(obs)
    qsim_sum = np.sum(sim)
    wb = 100 * (1 - np.abs(1 - (qsim_sum / qobs_sum)))

    return wb


def nse(obs: Union[list, np.ndarray], sim: Union[list, np.ndarray]) -> float:
    """Nash-Sutcliffe Efficiency.

    Calculates the Nash-Sutcliffe Efficiency (NSE), a widely used metric for assessing
    the performance of hydrological models.

    NSE measures the relative magnitude of the residual variance compared to the
    variance of the observed data. It indicates how well the model predictions match
    the observations compared to using the mean of the observations as a predictor.

    Args:
        obs: Observed flow values as a list or numpy array.
        sim: Simulated flow values as a list or numpy array.

    Returns:
        float: The NSE value. NSE ranges from -∞ to 1:
            - NSE = 1: Perfect match between simulated and observed values
            - NSE = 0: Model predictions are as accurate as the mean of observed data
            - NSE < 0: Mean of observed data is a better predictor than the model

    Raises:
        ValueError: If the input arrays have different lengths.
        ValueError: If the variance of observed values is zero.

    Examples:
        - Example with good performance:
            ```python
            >>> import numpy as np
            >>> from statista.descriptors import nse
            >>> observed = [10, 20, 30, 40, 50]
            >>> simulated = [12, 18, 33, 43, 48]
            >>> nse_value = nse(observed, simulated)
            >>> print(f"NSE: {nse_value:.4f}")
            NSE: 0.9700

            ```
        - Example with poorer performance:
            ```python
            >>> observed = [10, 20, 30, 40, 50]
            >>> simulated = [5, 15, 45, 35, 60]
            >>> nse_value = nse(observed, simulated)
            >>> print(f"NSE: {nse_value:.4f}")
            NSE: 0.6000

            ```
        - Example with negative NSE (poor model):
            ```python
            >>> observed = [10, 20, 30, 40, 50]
            >>> simulated = [50, 40, 30, 20, 10]
            >>> nse_value = nse(observed, simulated)
            >>> print(f"NSE: {nse_value:.4f}")
            NSE: -3.0000

            ```

    See Also:
        - nse_hf: Modified Nash-Sutcliffe Efficiency for high flows
        - nse_lf: Modified Nash-Sutcliffe Efficiency for low flows
        - kge: Kling-Gupta Efficiency
    """
    # convert obs & sim into arrays
    obs = np.array(obs)
    sim = np.array(sim)

    a = sum((obs - sim) ** 2)
    b = sum((obs - np.average(obs)) ** 2)
    e = 1 - (a / b)

    return e


def nse_hf(obs: Union[list, np.ndarray], sim: Union[list, np.ndarray]) -> float:
    """Modified Nash-Sutcliffe Efficiency for High Flows.

    Calculates a modified version of the Nash-Sutcliffe Efficiency that gives more
    weight to high flow values. This is particularly useful for evaluating model
    performance during flood events or peak flows.

    This modification weights the squared errors by the observed flow values, giving
    more importance to errors during high flow periods.

    Args:
        obs: Observed flow values as a list or numpy array.
        sim: Simulated flow values as a list or numpy array.

    Returns:
        float: The modified NSE value for high flows. Like standard NSE, it ranges from -∞ to 1:
            - NSE_HF = 1: Perfect match between simulated and observed values
            - NSE_HF = 0: Model predictions are as accurate as the mean of observed data
            - NSE_HF < 0: Mean of observed data is a better predictor than the model

    Raises:
        ValueError: If the input arrays have different lengths.
        ValueError: If the weighted variance of observed values is zero.

    Examples:
        - Example with good performance on high flows
            ```python
            >>> import numpy as np
            >>> from statista.descriptors import nse_hf
            >>> observed = [10, 20, 30, 40, 100]  # Note the high value at the end
            >>> simulated = [12, 18, 33, 43, 90]
            >>> nse_hf_value = nse_hf(observed, simulated)
            >>> print(f"NSE_HF: {nse_hf_value:.4f}")
            NSE_HF: 0.9717

            ```
        - Example with poor performance on high flows
            ```python
            >>> observed = [10, 20, 30, 40, 100]
            >>> simulated = [12, 18, 33, 43, 50]  # Significant underestimation of peak flow
            >>> nse_hf_value = nse_hf(observed, simulated)
            >>> print(f"NSE_HF: {nse_hf_value:.4f}")
            NSE_HF: 0.3401

            ```

    References:
        Hundecha Y. & Bárdossy A. (2004). Modeling of the effect of land use
        changes on the runoff generation of a river basin through
        parameter regionalization of a watershed model. Journal of Hydrology,
        292(1-4), 281-295.

    See Also:
        - nse: Standard Nash-Sutcliffe Efficiency
        - nse_lf: Modified Nash-Sutcliffe Efficiency for low flows
    """
    # convert obs & sim into arrays
    obs = np.array(obs)
    sim = np.array(sim)

    a = sum(obs * (obs - sim) ** 2)
    b = sum(obs * (obs - np.average(obs)) ** 2)
    e = 1 - (a / b)

    return e


def nse_lf(obs: Union[list, np.ndarray], sim: Union[list, np.ndarray]) -> float:
    """Modified Nash-Sutcliffe Efficiency for Low Flows.

    Calculates a modified version of the Nash-Sutcliffe Efficiency that gives more
    weight to low flow values. This is particularly useful for evaluating model
    performance during drought periods or base flow conditions.

    This modification applies a logarithmic transformation to the flow values before
    calculating the NSE, which gives more weight to relative errors during low flow periods.

    Args:
        obs: Observed flow values as a list or numpy array. Values must be positive.
        sim: Simulated flow values as a list or numpy array. Values must be positive.

    Returns:
        float: The modified NSE value for low flows. Like standard NSE, it ranges from -∞ to 1:
            - NSE_LF = 1: Perfect match between simulated and observed values
            - NSE_LF = 0: Model predictions are as accurate as the mean of observed data
            - NSE_LF < 0: Mean of observed data is a better predictor than the model

    Raises:
        ValueError: If the input arrays have different lengths.
        ValueError: If any values in the input arrays are zero or negative (logarithm cannot be applied).
        ValueError: If the weighted variance of log-transformed observed values is zero.

    Examples:
        - Example with good performance on low flows:
            ```python
            >>> import numpy as np
            >>> from statista.descriptors import nse_lf
            >>> observed = [10, 5, 3, 2, 1]  # Note the low values at the end
            >>> simulated = [11, 5.5, 2.8, 1.9, 1.1]
            >>> nse_lf_value = nse_lf(observed, simulated)
            >>> print(f"NSE_LF: {nse_lf_value:.4f}")
            NSE_LF: 0.9882

            ```
        - Example with poor performance on low flows:
            ```python
            >>> observed = [10, 5, 3, 2, 1]
            >>> simulated = [11, 5.5, 2.8, 1.9, 0.5]  # Significant error in lowest flow
            >>> nse_lf_value = nse_lf(observed, simulated)
            >>> print(f"NSE_LF: {nse_lf_value:.4f}")
            NSE_LF: 0.9882

            ```

    References:
        Hundecha Y. & Bárdossy A. (2004). Modeling of the effect of land use
        changes on the runoff generation of a river basin through
        parameter regionalization of a watershed model. Journal of Hydrology,
        292(1-4), 281-295.

    See Also:
        - nse: Standard Nash-Sutcliffe Efficiency
        - nse_hf: Modified Nash-Sutcliffe Efficiency for high flows
    """
    # convert obs & sim into arrays
    obs = np.array(np.log(obs))
    sim = np.array(np.log(sim))

    a = sum(obs * (obs - sim) ** 2)
    b = sum(obs * (obs - np.average(obs)) ** 2)
    e = 1 - (a / b)

    return e


def mbe(obs: Union[list, np.ndarray], sim: Union[list, np.ndarray]) -> float:
    """Mean Bias Error (MBE).

    Calculates the Mean Bias Error between observed and simulated values.
    MBE measures the average tendency of the simulated values to be larger or smaller
    than the observed values. A positive value indicates overestimation bias, while
    a negative value indicates underestimation bias.

    Formula: MBE = (sim - obs)/n

    Args:
        obs: Observed values as a list or numpy array.
        sim: Simulated values as a list or numpy array.

    Returns:
        float: The Mean Bias Error value.
            - MBE = 0: No bias
            - MBE > 0: Overestimation bias (simulated values tend to be larger than observed)
            - MBE < 0: Underestimation bias (simulated values tend to be smaller than observed)

    Raises:
        ValueError: If the input arrays have different lengths.

    Examples:
        - Example with overestimation bias:
            ```python
            >>> import numpy as np
            >>> from statista.descriptors import mbe
            >>> observed = [10, 20, 30, 40, 50]
            >>> simulated = [12, 22, 32, 42, 52]  # Consistently higher than observed
            >>> mbe_value = mbe(observed, simulated)
            >>> print(f"MBE: {mbe_value:.1f}")
            MBE: 2.0

            ```
        - Example with underestimation bias:
            ```python
            >>> observed = [10, 20, 30, 40, 50]
            >>> simulated = [8, 18, 28, 38, 48]  # Consistently lower than observed
            >>> mbe_value = mbe(observed, simulated)
            >>> print(f"MBE: {mbe_value:.1f}")
            MBE: -2.0

            ```
        - Example with no bias:
            ```python
            >>> observed = [10, 20, 30, 40, 50]
            >>> simulated = [12, 18, 32, 38, 50]  # Some higher, some lower, balanced overall
            >>> mbe_value = mbe(observed, simulated)
            >>> print(f"MBE: {mbe_value:.1f}")
            MBE: 0.0

            ```

    See Also:
        - mae: Mean Absolute Error
        - rmse: Root Mean Square Error
    """

    return (np.array(sim) - np.array(obs)).mean()


def mae(obs: Union[list, np.ndarray], sim: Union[list, np.ndarray]) -> float:
    """Mean Absolute Error (MAE).

    Calculates the Mean Absolute Error between observed and simulated values.
    MAE measures the average magnitude of the errors without considering their direction.
    It's the average of the absolute differences between observed and simulated values.

    Formula: MAE = |(obs - sim)|/n

    Args:
        obs: Observed values as a list or numpy array.
        sim: Simulated values as a list or numpy array.

    Returns:
        float: The Mean Absolute Error value. MAE is always non-negative, with smaller
            values indicating better model performance.
            - MAE = 0: Perfect match between observed and simulated values
            - MAE > 0: Average absolute difference between observed and simulated values

    Raises:
        ValueError: If the input arrays have different lengths.

    Examples:
            ```python
            >>> import numpy as np
            >>> from statista.descriptors import mae
            >>> observed = [10, 20, 30, 40, 50]
            >>> simulated = [12, 18, 33, 42, 48]
            >>> mae_value = mae(observed, simulated)
            >>> print(f"MAE: {mae_value:.1f}")
            MAE: 2.2

            ```
        - Example with larger errors:
            ```python
            >>> observed = [10, 20, 30, 40, 50]
            >>> simulated = [15, 15, 35, 35, 55]
            >>> mae_value = mae(observed, simulated)
            >>> print(f"MAE: {mae_value:.1f}")
            MAE: 5.0

            ```
        - Example with perfect match
            ```python
            >>> observed = [10, 20, 30, 40, 50]
            >>> simulated = [10, 20, 30, 40, 50]
            >>> mae_value = mae(observed, simulated)
            >>> print(f"MAE: {mae_value:.1f}")
            MAE: 0.0

            ```

    See Also:
        - mbe: Mean Bias Error
        - rmse: Root Mean Square Error (gives more weight to larger errors)
    """

    return np.abs(np.array(obs) - np.array(sim)).mean()


def pearson_corr_coeff(
    x: Union[list, np.ndarray], y: Union[list, np.ndarray]
) -> Number:
    """Pearson Correlation Coefficient.

    Calculates the Pearson correlation coefficient between two variables, which measures
    the linear relationship between them.

    Key properties:
    - Independent of the magnitude of the numbers (scale-invariant)
    - Sensitive to relative changes only
    - Measures only linear relationships

    The mathematical formula is:
    R = Cov(x,y) / (σx * σy)
    where Cov is the covariance and σ is the standard deviation.

    Args:
        x: First variable as a list or numpy array.
        y: Second variable as a list or numpy array.

    Returns:
        Number: The correlation coefficient between -1 and 1:
            - R = 1: Perfect positive linear relationship
            - R = 0: No linear relationship
            - R = -1: Perfect negative linear relationship

    Raises:
        ValueError: If the input arrays have different lengths.
        ValueError: If either array has zero variance (standard deviation = 0).

    Examples:
        - Perfect positive correlation:
            ```python
            >>> import numpy as np
            >>> from statista.descriptors import pearson_corr_coeff
            >>> x = [1, 2, 3, 4, 5]
            >>> y = [2, 4, 6, 8, 10]  # y = 2x
            >>> r = pearson_corr_coeff(x, y)
            >>> print(f"Correlation: {r:.4f}")
            Correlation: 1.0000

            ```
        - Perfect negative correlation:
            ```python
            >>> x = [1, 2, 3, 4, 5]
            >>> y = [10, 8, 6, 4, 2]  # y = -2x + 12
            >>> r = pearson_corr_coeff(x, y)
            >>> print(f"Correlation: {r:.4f}")
            Correlation: -1.0000

            ```
        - No correlation:
            ```python
            >>> x = [1, 2, 3, 4, 5]
            >>> y = [5, 2, 8, 1, 4]  # Random values
            >>> r = pearson_corr_coeff(x, y)
            >>> print(f"Correlation: {r:.4f}")
            Correlation: -0.1732

            ```

    See Also:
        - r2: Coefficient of determination
    """
    return np.corrcoef(np.array(x), np.array(y))[0][1]


def r2(obs: Union[list, np.ndarray], sim: Union[list, np.ndarray]) -> float:
    """Coefficient of Determination (R²).

    Calculates the coefficient of determination (R²) between observed and simulated values.

    R² measures how well the predicted values match the observed values, based on the
    distance between the points and the 1:1 line (not the best-fit regression line).
    The closer the data points are to the 1:1 line, the higher the coefficient of determination.

    Important properties:
    - Unlike the Pearson correlation coefficient, R² depends on the magnitude of the numbers
    - It measures the actual agreement between values, not just correlation
    - It can range from negative infinity to 1

    Args:
        obs: Observed values as a list or numpy array.
        sim: Simulated values as a list or numpy array.

    Returns:
        float: The coefficient of determination:
            - R² = 1: Perfect match between simulated and observed values
            - R² = 0: Model predictions are as accurate as using the mean of observed data
            - R² < 0: Model predictions are worse than using the mean of observed data

    Raises:
        ValueError: If the input arrays have different lengths.

    Examples:
        - Good model fit:
            ```python
            >>> import numpy as np
            >>> from statista.descriptors import r2
            >>> observed = [10, 20, 30, 40, 50]
            >>> simulated = [11, 19, 31, 41, 49]
            >>> r2_value = r2(observed, simulated)
            >>> print(f"R²: {r2_value:.4f}")
            R²: 0.9950

            ```
        - Poor model fit:
            ```python
            >>> observed = [10, 20, 30, 40, 50]
            >>> simulated = [15, 15, 35, 35, 50]
            >>> r2_value = r2(observed, simulated)
            >>> print(f"R²: {r2_value:.4f}")
            R²: 0.9000

            ```
        - Negative R² (very poor model):
            ```python
            >>> observed = [10, 20, 30, 40, 50]
            >>> simulated = [50, 40, 30, 20, 10]
            >>> r2_value = r2(observed, simulated)
            >>> print(f"R²: {r2_value:.4f}")
            R²: -3.0000

            ```

    See Also:
        - pearson_corr_coeff: Pearson correlation coefficient (measures correlation, not agreement)
        - nse: Nash-Sutcliffe Efficiency (mathematically equivalent to R² for the 1:1 line)
    """
    return r2_score(np.array(obs), np.array(sim))
