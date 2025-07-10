import polars as pl
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

df = pl.read_excel("data/CASchools2.xlsx").sort("avginc")


def func(income, b0, b1, b2):
    return b0 * (1 - np.exp(-b1 * (income - b2)))


initial_guess = [250, 0.1, 5.0]
params2, covariance = curve_fit(func, df["avginc"], df["testscr"], p0=initial_guess)

plt.scatter(df["avginc"], df["testscr"], label="Observed Data", color="blue", alpha=0.6)
plt.plot(
    df["avginc"],
    func(df["avginc"], *params2),
    label="Fitted Model",
    color="red",
    linewidth=2,
)
plt.xlabel("avginc")
plt.ylabel("testscr")
plt.title("Nonlinear Fit: testscr = b0 * (1 - exp(-b1 * (avginc - b2)))")
plt.legend()
plt.grid(True)
plt.savefig("assets/fig4.png")

