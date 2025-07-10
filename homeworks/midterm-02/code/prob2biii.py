coeffs = model.params[["attend_v", "attend_m", "attend_n"]].values
cov = (
    model.cov_params()
    .loc[["attend_v", "attend_m", "attend_n"], ["attend_v", "attend_m", "attend_n"]]
    .values
)

delta_x = np.array([6, -2, -1])

delta_ln_assaults = np.dot(delta_x, coeffs)

std_error = np.sqrt(np.dot(delta_x, np.dot(cov, delta_x)))

lower = delta_ln_assaults - 1.96 * std_error
upper = delta_ln_assaults + 1.96 * std_error

percent_change = 100 * (np.exp(delta_ln_assaults) - 1)
ci_lower = 100 * (np.exp(lower) - 1)
ci_upper = 100 * (np.exp(upper) - 1)

print(f"Predicted % change in assaults: {percent_change:.2f}%")
print(f"95% CI: [{ci_lower:.2f}%, {ci_upper:.2f}%]")
