import numpy as np
from scipy import stats


def main() -> None:
    e_e = 520
    n = 29
    k = 3
    XTX = np.array([[29, 0, 0], [0, 50, 10], [0, 10, 80]])
    beta_hat = np.array([4, 0.4, 0.9])

    sigma_squared = e_e / (n - k)

    XTX_inv = np.linalg.inv(XTX)

    R = np.array([[0, 1, 1]])
    r = 1

    Rb_minus_r = R @ beta_hat - r
    denominator = R @ XTX_inv @ R.T * sigma_squared
    F_stat = (Rb_minus_r**2) / denominator

    df1 = 1
    df2 = n - k

    p_value = 1 - stats.f.cdf(F_stat, df1, df2)

    print("F-statistic:", F_stat[0][0])
    print("p-value:", p_value)


if __name__ == "__main__":
    main()
