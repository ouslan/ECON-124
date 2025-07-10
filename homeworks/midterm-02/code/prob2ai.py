import polars as pl
import numpy as np
import statsmodels.formula.api as smf
from linearmodels.iv import IV2SLS

df = pl.read_excel("data/Movies.xlsx")
df = df.select(
    pl.all().exclude("month1", "year1"),
    ln_assaults=pl.col("assaults").log(),
    attendance=pl.col("attend_v") + pl.col("attend_m") + pl.col("attend_n"),
)
data = df.to_pandas()

variables = df.select(
    pl.all().exclude(
        "assaults", "ln_assaults", "^atten.*$", "^h_.*$", "^pr_.*$", "^w.*$"
    )
).columns
formula = "ln_assaults ~ " + " + ".join(variables)


model = smf.ols(formula=formula, data=df).fit()
print(model.summary())
