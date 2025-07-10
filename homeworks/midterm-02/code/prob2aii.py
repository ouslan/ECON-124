## assumes the code in 2ai ran
variables = df.select(
    pl.all().exclude(
        "assaults", "ln_assaults", "^atten.*$", "^h_.*$", "^pr_.*$", "^w.*$"
    )
).columns
formula = "attendance ~ " + " + ".join(variables)


model = smf.ols(formula=formula, data=df).fit()
print(model.summary())

