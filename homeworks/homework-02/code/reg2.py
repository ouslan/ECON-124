import numpy as np
import polars as pl
import statsmodels.api as sm

df = pl.read_excel("data/cps09mar.xlsx")
df = df.filter((pl.col("female") == 0) & (pl.col("hisp") == 1))
df = df.with_columns(
    north_dummy=pl.when(pl.col("region") == 1).then(1).otherwise(0),
    south_dummy=pl.when(pl.col("region") == 3).then(1).otherwise(0),
    west_dummy=pl.when(pl.col("region") == 4).then(1).otherwise(0),
    married_dummy=pl.when(
        (pl.col("marital") == 1) | (pl.col("marital") == 2) | (pl.col("marital") == 3)
    )
    .then(1)
    .otherwise(0),
    widowed_dummy=pl.when((pl.col("marital") == 4)).then(1).otherwise(0),
    seperated_dummy=pl.when((pl.col("marital") == 5) | (pl.col("marital") == 6))
    .then(1)
    .otherwise(0),
    wage=(pl.col("earnings") / (pl.col("hours") * pl.col("week"))).log(),
    exp=(pl.col("age") - pl.col("education")),
    exp_2=(pl.col("age") - pl.col("education")) ** 2,
).to_pandas()
Y = df["wage"].values.reshape(-1, 1)
X = df[
    [
        "education",
        "north_dummy",
        "south_dummy",
        "west_dummy",
        "married_dummy",
        "widowed_dummy",
        "seperated_dummy",
        "wage",
        "exp",
        "exp_2",
    ]
].values.reshape(-1, 10)
X = sm.add_constant(X)
betas = np.linalg.pinv(X.T @ X) @ X.T @ Y

print(betas)
