import pandas as pd

r_dir = r"C:\gdrive\papers\MyPapers\60-Heavy-tail-statistics"
# %%
shape_1d = pd.read_csv(f"{r_dir}/shape_1d_lmoment.csv", index_col=0)
shape_1d.drop(columns=["gauges"], inplace=True)
shape_2d = pd.read_csv(f"{r_dir}/shape_2d_lmoment.csv", index_col=0)
shape_2d.drop(columns=["gauges"], inplace=True)
# %
median_1d = shape_1d.median(axis=1)
median_2d = shape_2d.median(axis=1)
median_diff = median_1d - median_2d

q75_1d = shape_1d.quantile(
    q=0.75, axis=1, numeric_only=True, interpolation="midpoint", method="single"
)
q25_1d = shape_1d.quantile(
    q=0.25, axis=1, numeric_only=True, interpolation="midpoint", method="single"
)
iqr_1d = q75_1d - q25_1d


q75_2d = shape_2d.quantile(
    q=0.75, axis=1, numeric_only=True, interpolation="midpoint", method="single"
)
q25_2d = shape_2d.quantile(
    q=0.25, axis=1, numeric_only=True, interpolation="midpoint", method="single"
)
iqr_2d = q75_2d - q25_2d
iqr_diff = iqr_1d - iqr_2d
# %%

mean_1d = shape_1d.mean(axis=1)
mean_2d = shape_2d.mean(axis=1)
mean_diff = mean_1d - mean_2d
