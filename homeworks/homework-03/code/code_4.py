import polars as pl
import statsmodels.api as sm
import numpy as np

from scipy.stats import chi2


def main() -> None:
    df = pl.read_excel("data/Nerlove1963.xlsx").to_pandas()
    for col in df.columns:
        df[f"{col}_log"] = np.log(df[col])
    X = df[["output_log", "Plabor_log", "Pcapital_log", "Pfuel_log"]]
    X = sm.add_constant(X)
    y = df["Cost_log"]

    model = sm.OLS(y, X).fit()

    print(model.summary())

    # 4c
    glm_model = sm.GLM(y, X, family=sm.families.Gaussian())
    constraint = "Plabor_log + Pcapital_log + Pfuel_log = 1"

    model_constrained = glm_model.fit_constrained(constraint)
    print(model_constrained.summary())
    # 4d
    R = np.array([[0, 0, 1, 1, 1]])
    q = np.array([1])

    beta_hat = model_constrained.params.values
    cov_beta = model_constrained.cov_params().values

    W = (R @ beta_hat - q) @ np.linalg.inv(R @ cov_beta @ R.T) @ (R @ beta_hat - q)
    p_value = 1 - chi2.cdf(W, df=1)

    print(f"Wald statistic: {W:.4f}")
    print(f"p-value: {p_value:.4f}")


if __name__ == "__main__":
    main()
