# aasums the code in 2 A i ran
variables = df.select(
    pl.all().exclude("assaults", "ln_assaults", "wkd_ind", "^pr_.*$", "^atten.*$")
).columns
formula = "ln_assaults ~ " + "attend_v + attend_m + attend_n + " + " + ".join(variables)

model = smf.ols(formula=formula, data=df).fit()
print(model.summary())
