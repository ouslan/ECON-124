import arviz as az
import numpy as np
import pandas as pd
import bambi as bmb
from sklearn.linear_model import LinearRegression

az.style.use("arviz-darkgrid")


def mc_func(size, intercept, slope, SEED=787):
    rng = np.random.default_rng(SEED)
    x = np.linspace(0, 1, size)
    true_regression_line = intercept + slope * x
    y = true_regression_line + rng.normal(scale=3, size=size)
    data = pd.DataFrame({"x": x, "y": y})
    model = bmb.Model("y ~ x", data)
    idata = model.fit(draws=1000, chains=1)
    return idata


def asymptotic_variance(size, intercept, slope, seed=787):
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 1, size)
    true_regression_line = intercept + slope * x
    y = true_regression_line + rng.normal(scale=3, size=size)
    data = pd.DataFrame({"x": x, "y": y})

    model = LinearRegression().fit(data[["x"]], data["y"])
    residuals = data["y"] - model.predict(data[["x"]])
    sigma_sq = np.var(residuals, ddof=1)
    x_centered = data["x"] - data["x"].mean()
    var_beta1 = sigma_sq / np.sum(x_centered**2)

    return var_beta1


def bootstrap_slope_variance(size, intercept, slope, n_bootstrap=1000, seed=787):
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 1, size)
    true_regression_line = intercept + slope * x
    y = true_regression_line + rng.normal(scale=3, size=size)
    data = pd.DataFrame({"x": x, "y": y})

    coefs = []
    for _ in range(n_bootstrap):
        sample = data.sample(n=size, replace=True, random_state=rng.integers(1e6))
        model = LinearRegression().fit(sample[["x"]], sample["y"])
        coefs.append(model.coef_[0])

    return np.var(coefs)


true_intercept = 1
true_slope = 2

asymp_var_25 = asymptotic_variance(25, true_intercept, true_slope)
asymp_var_50 = asymptotic_variance(50, true_intercept, true_slope)
asymp_var_100 = asymptotic_variance(100, true_intercept, true_slope)

boot_var_25 = bootstrap_slope_variance(25, true_intercept, true_slope)
boot_var_50 = bootstrap_slope_variance(50, true_intercept, true_slope)
boot_var_100 = bootstrap_slope_variance(100, true_intercept, true_slope)

idata1 = mc_func(size=25, slope=2, intercept=1)
idata2 = mc_func(size=50, slope=2, intercept=1)
idata3 = mc_func(size=100, slope=2, intercept=1)

posterior_slope_25 = idata1.posterior["x"].values.flatten()
posterior_slope_50 = idata2.posterior["x"].values.flatten()
posterior_slope_100 = idata3.posterior["x"].values.flatten()

posterior_var_25 = posterior_slope_25.var()
posterior_var_50 = posterior_slope_50.var()
posterior_var_100 = posterior_slope_100.var()

az.plot_density(
    [idata1, idata2, idata3],
    var_names=["x"],
    data_labels=["25", "50", "100"],
    shade=0.2,
)

df = pd.DataFrame(
    {
        "Sample Size": [25, 50, 100],
        "Posterior Var": [posterior_var_25, posterior_var_50, posterior_var_100],
        "Asymptotic Var": [asymp_var_25, asymp_var_50, asymp_var_100],
        "Bootstrap Var": [boot_var_25, boot_var_50, boot_var_100],
    }
)

print(df)
