# Time Series Subpackage

The `statista.time_series` subpackage provides the `TimeSeries` class — a pandas
`DataFrame` subclass with **53 statistical analysis methods** across 12 functional
categories, designed for researchers in hydrology, climate science, and environmental
engineering.

```python
from statista.time_series import TimeSeries
```

## Architecture

`TimeSeries` is composed from 12 parent classes (mixins), each providing a specific
category of functionality. At runtime, they all merge into a single class that extends
`pandas.DataFrame`.

```mermaid
classDiagram
    class DataFrame {
        <<pandas>>
    }
    class TimeSeriesBase {
        +__init__(data, index, columns)
        +_get_ax_fig()
        +_adjust_axes_labels()
    }
    class Descriptive {
        +stats
        +extended_stats
        +l_moments()
        +summary()
    }
    class Visualization {
        +box_plot()
        +violin()
        +raincloud()
        +histogram()
        +density()
        +rolling_statistics()
    }
    class MissingData {
        +missing_summary()
        +gap_analysis()
        +completeness_report()
        +detect_outliers()
        +outlier_plot()
    }
    class Correlation {
        +acf()
        +pacf()
        +cross_correlation()
        +lag_plot()
        +correlation_matrix()
        +ljung_box()
    }
    class Stationarity {
        +adf_test()
        +kpss_test()
        +stationarity_summary()
    }
    class Trend {
        +mann_kendall()
        +sens_slope()
        +detrend()
        +innovative_trend_analysis()
    }
    class Distribution {
        +qq_plot()
        +pp_plot()
        +normality_test()
        +empirical_cdf()
        +fit_distributions()
    }
    class ChangePoint {
        +pettitt_test()
        +snht_test()
        +buishand_range_test()
        +cusum()
        +homogeneity_summary()
    }
    class Decomposition {
        +classical_decompose()
        +smooth()
        +envelope()
    }
    class Seasonal {
        +monthly_stats()
        +seasonal_subseries()
        +annual_cycle()
        +periodogram()
        +seasonal_mann_kendall()
    }
    class Hydrological {
        +flow_duration_curve()
        +annual_extremes()
        +exceedance_probability()
        +baseflow_separation()
        +baseflow_index()
        +flashiness_index()
        +recession_analysis()
    }
    class Comparison {
        +anomaly()
        +standardized_anomaly()
        +double_mass_curve()
        +regime_comparison()
    }
    class TimeSeries {
        +_constructor
    }

    DataFrame <|-- TimeSeriesBase
    TimeSeriesBase <|-- TimeSeries
    Descriptive <|-- TimeSeries
    Visualization <|-- TimeSeries
    MissingData <|-- TimeSeries
    Correlation <|-- TimeSeries
    Stationarity <|-- TimeSeries
    Trend <|-- TimeSeries
    Distribution <|-- TimeSeries
    ChangePoint <|-- TimeSeries
    Decomposition <|-- TimeSeries
    Seasonal <|-- TimeSeries
    Hydrological <|-- TimeSeries
    Comparison <|-- TimeSeries
```

## Module Dependencies

Internal cross-module imports between the time series subpackage files:

```mermaid
graph TD
    subgraph "statista.time_series"
        INIT["__init__.py<br/>TimeSeries"]
        BASE["base.py<br/>TimeSeriesBase"]
        DESC["descriptive.py"]
        VIZ["visualization.py"]
        MISS["missing.py"]
        CORR["correlation.py"]
        STAT["stationarity.py"]
        TREND["trend.py"]
        DIST["distribution.py"]
        CP["changepoint.py"]
        DECOMP["decomposition.py"]
        SEAS["seasonal.py"]
        HYDRO["hydrological.py"]
        COMP["comparison.py"]
    end

    INIT --> BASE
    INIT --> DESC
    INIT --> VIZ
    INIT --> MISS
    INIT --> CORR
    INIT --> STAT
    INIT --> TREND
    INIT --> DIST
    INIT --> CP
    INIT --> DECOMP
    INIT --> SEAS
    INIT --> HYDRO
    INIT --> COMP
    VIZ --> BASE
    TREND --> CORR
    SEAS --> TREND

    subgraph "External"
        NP["numpy"]
        SP["scipy.stats<br/>scipy.signal"]
        MPL["matplotlib"]
        PD["pandas"]
    end

    BASE --> PD
    BASE --> MPL
    DESC --> SP
    CORR --> SP
    STAT --> SP
    TREND --> SP
    DIST --> SP
    CP --> SP
    DECOMP --> SP
    SEAS --> SP
    HYDRO --> PD
    COMP --> SP
```

## Typical Analysis Workflow

```mermaid
flowchart LR
    A[Load Data] --> B[Quality Check]
    B --> C{Stationary?}
    C -->|Yes| D[Trend Analysis]
    C -->|No| E[Detrend / Difference]
    E --> D
    D --> F[Distribution Fitting]
    F --> G[Return Periods]
    D --> H[Seasonal Analysis]
    D --> I[Change Point Detection]

    B --> B1[missing_summary]
    B --> B2[detect_outliers]
    B --> B3[completeness_report]
    C --> C1[stationarity_summary]
    C --> C2[acf / pacf]
    D --> D1[mann_kendall]
    D --> D2[sens_slope]
    F --> F1[normality_test]
    F --> F2[fit_distributions]
    F --> F3[qq_plot]
    I --> I1[homogeneity_summary]
    H --> H1[monthly_stats]
    H --> H2[periodogram]

    style A fill:#e1f5fe
    style G fill:#c8e6c9
```

## Hydrological Analysis Pipeline

```mermaid
flowchart TD
    RAW[Raw Streamflow Data] --> QC[Quality Control]
    QC --> FDC[Flow Duration Curve]
    QC --> BFS[Baseflow Separation]
    QC --> AMS[Annual Maxima]

    BFS --> BFI[Baseflow Index]
    BFS --> FI[Flashiness Index]

    AMS --> EVA[Extreme Value Analysis<br/>fit_distributions]
    EVA --> RP[Return Periods<br/>exceedance_probability]

    QC --> REC[Recession Analysis]
    QC --> SEAS[Seasonal Analysis<br/>monthly_stats]
    QC --> TREND[Trend Detection<br/>mann_kendall]
    QC --> CP[Change Point<br/>homogeneity_summary]

    style RAW fill:#bbdefb
    style RP fill:#c8e6c9
    style BFI fill:#c8e6c9
    style FI fill:#c8e6c9
```

## Method Categories

| Category | Module | Methods | Purpose |
|---|---|---|---|
| [Descriptive](descriptive.md) | `descriptive.py` | 4 | Summary statistics, L-moments |
| [Visualization](visualization.md) | `visualization.py` | 7 | Box, violin, raincloud, histogram, KDE |
| [Missing Data](missing.md) | `missing.py` | 5 | Gap analysis, outlier detection |
| [Correlation](correlation.md) | `correlation.py` | 6 | ACF, PACF, cross-correlation, Ljung-Box |
| [Stationarity](stationarity.md) | `stationarity.py` | 3 | ADF, KPSS, combined diagnosis |
| [Trend](trend.md) | `trend.py` | 4 | Mann-Kendall (5 variants), Sen's slope |
| [Distribution](distribution.md) | `distribution.py` | 5 | QQ/PP plots, normality tests |
| [Change Point](changepoint.md) | `changepoint.py` | 5 | Pettitt, SNHT, Buishand, CUSUM |
| [Decomposition](decomposition.md) | `decomposition.py` | 3 | Classical decompose, smoothing |
| [Seasonal](seasonal.md) | `seasonal.py` | 5 | Monthly stats, periodogram |
| [Hydrological](hydrological.md) | `hydrological.py` | 7 | FDC, baseflow, recession |
| [Comparison](comparison.md) | `comparison.py` | 4 | Anomaly, regime comparison |

## Quick Start

```python
import numpy as np
from statista.time_series import TimeSeries

# Create from data
data = np.loadtxt("examples/data/time_series1.txt")
ts = TimeSeries(data)

# Descriptive statistics
ts.extended_stats
ts.summary()

# Stationarity + Trend
ts.stationarity_summary()
ts.mann_kendall(method="hamed_rao")

# Distribution fitting
ts.normality_test()
ts.qq_plot()
```
