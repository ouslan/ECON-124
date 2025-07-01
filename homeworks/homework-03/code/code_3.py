import polars as pl
import statsmodels.api as sm


def main():
    df1 = pl.read_excel("data/timeinvar.xlsx")
    df2 = pl.read_excel("data/timevar.xlsx")
    df = df1.join(df2, on="id", how="left", validate="m:m")
    # 3b

    X_1 = df[["edu", "exper", "ability"]].to_numpy()
    X_1 = sm.add_constant(X_1)
    y = df[["lwage"]].to_numpy()
    res1 = sm.OLS(y, X_1).fit()
    print(res1.summary())

    X_2 = df[["meduc", "feduc", "brokenhome", "siblings"]].to_numpy()
    X_2 = sm.add_constant(X_2)
    y = df[["lwage"]].to_numpy()
    res2 = sm.OLS(y, X_2).fit()
    print(res2.summary())

    # 3b
    A = np.identity(len(res1.params))
    A = A[1:, :]

    f_test_result = res1.f_test(A)
    print(f_test_result)

    # 3c
    A = np.identity(len(res2.params))
    A = A[1:, :]

    f_test_result = res2.f_test(A)
    print(f_test_result)

    # 3d
    res2.wald_test(A)


if __name__ == "__main__":
    main()
