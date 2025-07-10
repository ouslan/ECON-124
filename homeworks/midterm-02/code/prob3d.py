formula  = "lwage ~ edu + edu2 + educab + ability + exper + meduc + feduc + brokenhome + siblings"

model = smf.ols(formula=formula, data=data).fit(cov_type='HC1')

print(model.summary())

mean_ability = data['ability'].mean()
low_ability_data = data[data['ability'] < mean_ability]
high_ability_data = data[data['ability'] >= mean_ability]

low_ability_lwage = model.predict(low_ability_data)
high_ability_lwage = model.predict(high_ability_data)

plt.scatter(low_ability_data['edu'], low_ability_lwage, label='Low Ability', color='blue')

plt.scatter(high_ability_data['edu'], high_ability_lwage, label='High Ability', color='red')

plt.xlabel('Years of Education')
plt.ylabel('Predicted Log Wage')
plt.savefig("assets/fig3.png")