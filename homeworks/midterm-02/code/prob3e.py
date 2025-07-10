formula = (
    "lwage ~ tao1 + tao2 + ability + exper + meduc + feduc + brokenhome + siblings"
)

model = smf.ols(formula=formula, data=data).fit()
print(model.summary())

