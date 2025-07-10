endog = data[['attend_v']]
other_controls  = df.select(pl.all().exclude("assaults", "ln_assaults", "wkd_ind", "^atten.*$", "^pr_.*$")).columns
exog = data[['attend_m', 'attend_n'] + other_controls] 
instruments = data[["attend_v_f", "attend_m_f", "attend_n_f", "attend_v_b", "attend_m_b", "attend_n_b"]]
ivolsmod = IV2SLS(dependent=data[["ln_assaults"]], endog=endog, exog=exog, instruments=instruments)
res_ivols = ivolsmod.fit()
print(res_ivols.summary)