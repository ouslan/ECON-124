import polars as pl
import numpy as np
from scipy.optimize import minimize
from scipy.stats import rv_continuous

df = pl.read_excel("data/Consumption.xlsx").to_pandas()
C = df["realcons"].values
Y = df["realgdp"].values


class consumption_model(rv_continuous):
    def __init__(self, Y, C):
        super().__init__(name="consumption")
        self.Y = np.asarray(Y)
        self.C = np.asarray(C)
        self.n = len(C)

    def _loglike(self, params):
        alpha, beta, gamma, sigma_squared = params
        predicted_C = alpha + beta * self.Y**gamma
        residuals = self.C - predicted_C
        ll = (
            -self.n / 2 * np.log(sigma_squared)
            - np.log(2 * np.pi)
            - (1 / (2 * sigma_squared)) * np.sum(residuals**2)
        )
        return -ll

    def fit(self, start_params=None, bounds=None):
        if start_params is None:
            start_params = [50, 2, 1.2, 1]

        if bounds is None:
            bounds = [(None, None), (None, None), (1e-6, None), (1e-6, None)]

        result = minimize(self._loglike, start_params, method="L-BFGS-B", bounds=bounds)
        self.mle_result = result

        if result.success:
            self.alpha_hat, self.beta_hat, self.gamma_hat, self.sigma2_hat = result.x
        else:
            raise RuntimeError("MLE optimization failed.")

        return result.x


model = consumption_model(Y, C)
alpha_hat, beta_hat, gamma_hat, sigma2_hat = model.fit()

print(f"Estimated alpha: {alpha_hat:.4f}")
print(f"Estimated beta: {beta_hat:.4f}")
print(f"Estimated gamma: {gamma_hat:.4f}")
print(f"Estimated sigma^2: {sigma2_hat:.4f}")
