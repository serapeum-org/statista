import datetime as dt

import matplotlib.pyplot as plt
import pandas as pd

date_format = "%Y-%m-%d"
rdir = rf"examples/data/rhine-2.csv"
# %%
df = pd.read_csv(rdir)
start = dt.datetime.strptime(df.loc[0, "date"], date_format)
end = dt.datetime.strptime(df.loc[df.index[-1], "date"], date_format)
ind = pd.date_range(start, end)
df.index = ind
df.drop(labels=["date"], axis=1, inplace=True)
df.plot()
plt.show()
# %%
from statista.eva import ams_analysis

statistical_properties, distribution_properties = ams_analysis(
    df,
    save_plots=True,
    save_to=f"{rdir}/lmoments",
    filter_out=0,
    method="lmoments",
    significance_level=0.05,
)
