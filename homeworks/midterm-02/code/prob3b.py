# b

formula  = "lwage ~ edu + edu2 + ability + exper + meduc + feduc + brokenhome + siblings"


model = smf.ols(formula=formula, data=data).fit()
print(model.summary())

plt.scatter(data["edu"], data["lwage"])
plt.savefig("assets/fig1.png")

plt.hist(data["edu"])
plt.savefig("assets/fig2.png")
