import polars as pl
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import numpy as np

# Question 3

df1 = pl.read_excel("data/timeinvar-1.xlsx")
df2 = pl.read_excel("data/timevar-1.xlsx")
df = df1.join(df2, on="id", how="left", validate="1:m")
df = df.with_columns(
    hs=pl.when(pl.col("edu") <= 12).then(1).otherwise(0),
    col=pl.when((pl.col("edu") > 12) & (pl.col("edu") <= 16)).then(1).otherwise(0),
    tao1=pl.when((pl.col("edu") <= 16)).then(pl.col("edu")).otherwise(0),
    tao2=pl.when((pl.col("edu") > 16)).then(pl.col("edu")).otherwise(0),
    grad=pl.when(pl.col("edu") > 16).then(1).otherwise(0),
    educab=pl.col("edu") * pl.col("ability"),
    edu2=pl.col("edu") ** 2,
)
data = df.to_pandas()

formula = "lwage ~ edu + ability + exper + meduc + feduc + brokenhome + siblings"


model = smf.ols(formula=formula, data=data).fit()
print(model.summary())

formula = "lwage ~ col + grad + ability + exper + meduc + feduc + brokenhome + siblings"


model = smf.ols(formula=formula, data=data).fit()
print(model.summary())

