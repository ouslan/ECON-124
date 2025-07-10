# c

formula = (
    "lwage ~ edu + educab + ability + exper + meduc + feduc + brokenhome + siblings"
)

model = smf.ols(formula=formula, data=data).fit(cov_type="HC1")

print(model.summary())


coeffs = model.params[["edu", "educab"]].values
cov = model.cov_params().loc[["edu", "educab"], ["edu", "educab"]].values

delta_x = np.array([1, 0.052374])

delta_ln_assaults = np.dot(delta_x, coeffs)

std_error = np.sqrt(np.dot(delta_x, np.dot(cov, delta_x)))

lower = delta_ln_assaults - 1.96 * std_error
upper = delta_ln_assaults + 1.96 * std_error

change = np.exp(delta_ln_assaults) - 1
ci_lower = np.exp(lower) - 1
ci_upper = np.exp(upper) - 1

print(f"Predicted change in assaults: {change:.2f}")
print(f"95% CI: [{ci_lower:.2f}, {ci_upper:.2f}]")

