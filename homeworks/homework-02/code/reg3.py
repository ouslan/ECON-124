import numpy as np
import polars as pl
import statsmodels.api as sm

df = pl.read_excel("data/Koop_Tobias_subsample.xlsx").to_pandas()

Y = df["lwage"].values.reshape(-1, 1)
X_1 = df[["Education", "Experience", "Ability"]].values.reshape(-1, 3)
X_1 = sm.add_constant(X_1)

n = len(Y)
p = 3

beta_1 = np.linalg.pinv(X_1.T @ X_1) @ X_1.T @ Y

y_pred = X_1 @ beta_1

ss_res = np.sum((Y - y_pred) ** 2)
ss_tot = np.sum((Y - np.mean(Y)) ** 2)
r_squared = 1 - ss_res / ss_tot

r_squared_adj = 1 - ((1 - r_squared) * (n - 1)) / (n - p - 1)

print("Model 1")
print("R:", r_squared)
print(beta_1)
print("Adjusted R:", r_squared_adj)

Y = df["lwage"].values.reshape(-1, 1)
X_1 = df[["Mothers_Educ", "Fathers_Educ", "Siblings"]].values.reshape(-1, 3)

n = len(Y)
p = 3

beta_1 = np.linalg.pinv(X_1.T @ X_1) @ X_1.T @ Y

y_pred = X_1 @ beta_1

ss_res = np.sum((Y - y_pred) ** 2)
ss_tot = np.sum((Y - np.mean(Y)) ** 2)
r_squared = 1 - ss_res / ss_tot

r_squared_adj = 1 - ((1 - r_squared) * (n - 1)) / (n - p - 1)

print("model 2")
print("R:", r_squared)
print(beta_1)
print("Adjusted R:", r_squared_adj)


data = df.to_pandas()
Y = data["lwage"].values.reshape(-1, 1)
X_1 = data[
    ["Education", "Experience", "Ability", "Mothers_Educ", "Fathers_Educ", "Siblings"]
].values.reshape(-1, 6)

n = len(Y)
p = 6

beta_1 = np.linalg.pinv(X_1.T @ X_1) @ X_1.T @ Y

y_pred = X_1 @ beta_1

ss_res = np.sum((Y - y_pred) ** 2)
ss_tot = np.sum((Y - np.mean(Y)) ** 2)
r_squared = 1 - ss_res / ss_tot

r_squared_adj = 1 - ((1 - r_squared) * (n - 1)) / (n - p - 1)

print("full model")
print("R:", r_squared)
print(beta_1)
print("Adjusted R:", r_squared_adj)


residuals = []  # c
X1_vars = [
    "Education",
    "Experience",
    "Ability",
]
X2_vars = [
    "Mothers_Educ",
    "Fathers_Educ",
    "Siblings",
]


X1 = sm.add_constant(df[X1_vars].values.reshape(-1, 3))

for var in X2_vars:
    Y = df[var].values.reshape(-1, 1)
    bata_hat = np.linalg.pinv(X1.T @ X1) @ X1.T @ Y
    res = Y - X1 @ bata_hat
    residuals.append(res)

X2_star2 = np.column_stack(residuals)
print(X2_star2)
