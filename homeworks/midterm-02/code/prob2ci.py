endog = data[["attend_v"]]
other_controls = df.select(
    pl.all().exclude("assaults", "ln_assaults", "wkd_ind", "^atten.*$", "^pr_.*$")
).columns
exog = data[["attend_m", "attend_n"] + other_controls]
instruments = data[["pr_attend_v", "pr_attend_m", "pr_attend_n"]]
ivolsmod = IV2SLS(
    dependent=data[["ln_assaults"]], endog=endog, exog=exog, instruments=instruments
)
res_ivols = ivolsmod.fit()
print(res_ivols.summary)

