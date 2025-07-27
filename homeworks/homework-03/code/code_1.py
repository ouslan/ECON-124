import numpy as np
from numpy.linalg import inv

XTX = np.array(
    [
        [100, 123, 96, 109],
        [123, 252, 125, 189],
        [96, 125, 167, 146],
        [109, 189, 146, 168],
    ]
)

XTy = np.array([460, 810, 615, 712]).reshape(-1, 1)

yTy = 3924


def main() -> None:
    # Problem 1a
    M = np.array([[252, 125, 189], [125, 167, 146], [189, 146, 168]])
    std_devs = np.sqrt(np.diag(M))

    correlation_matrix = M / np.outer(std_devs, std_devs)
    print(np.round(correlation_matrix, 4))

    # Problem 1b
    print(inv(XTX) @ XTy)

    # Problem 1c
    XTX_1 = np.delete(np.delete(XTX, 3, 0), 3, 1)
    XTy_1 = np.delete(XTy, 3, 0)
    print(inv(XTX_1) @ XTy_1)

    XTX_2 = np.delete(np.delete(XTX, 2, 0), 2, 1)
    XTy_2 = np.delete(XTy, 2, 0)
    print(inv(XTX_2) @ XTy_2)

    XTX_3 = np.delete(np.delete(XTX, 1, 0), 1, 1)
    XTy_3 = np.delete(XTy, 1, 0)
    print(inv(XTX_3) @ XTy_3)

    # Problem 1d
    XTX_no_const = np.delete(np.delete(XTX, 0, 0), 0, 1)

    stds = np.sqrt(np.diag(XTX_no_const))

    R = XTX_no_const / np.outer(stds, stds)

    R_inv = np.linalg.inv(R)
    VIFs = np.diag(R_inv)

    for i, vif in enumerate(VIFs, start=1):
        print(f"VIF for x_{i}: {vif:.2f}")


if __name__ == "__main__":
    main()
