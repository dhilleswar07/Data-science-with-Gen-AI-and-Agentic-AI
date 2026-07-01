
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv(r"E:\DATASCIENCE WITH GEN AI & AGENTIC AI\Salary_Data.csv")

x= dataset.iloc[:,:-1]
y= dataset.iloc[:, -1]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.20,train_size=0.80,random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)

y_pred = regressor.predict(x_test)
y_pred = regressor.predict(x_test)
plt.scatter(x_test, y_test,color='red')
plt.plot(x_train, regressor.predict(x_train), color= 'blue')
plt.title('salary vs experence ')
plt.xlabel('year of experience')
plt.ylabel('salary')
plt.show()

dataset

m_coef = regressor.coef_
print(m_coef)
c_intercept = regressor.intercept_
print(c_intercept)
y_12 = m_coef * 12 + c_intercept
print(y_12)

bias_score = regressor.score(x_train,y_train)
print(bias_score)

variance_score = regressor.score(x_test,y_test)
print(variance_score)

# Statistics integration

dataset.mean()
dataset['Salary'].mean()
dataset['YearsExperience'].mean()

dataset.median()
dataset['Salary'].median()
dataset['YearsExperience'].median()

dataset.var()
dataset['Salary'].var()
dataset['YearsExperience'].var()

dataset.std()
dataset['Salary'].std()
dataset['YearsExperience'].std()

from scipy.stats import variation
variation(dataset.values)
variation(dataset['Salary'])
variation(dataset['YearsExperience'])

dataset.corr()

dataset['Salary'].corr(dataset['YearsExperience'])
dataset['YearsExperience'].corr(dataset['Salary'])

dataset.skew()

dataset['Salary'].skew()

dataset.sem()

dataset['Salary'].sem()

import scipy.stats as stats
dataset.apply(stats.zscore) # this will give z-score for entire dataset

stats.zscore(dataset['Salary']) # this will give z-score for praticular column

# ANOVA

y_mean=np.mean(y)
SSR = np.sum((y_pred-y_mean)**2)
print(SSR)

y=y[0:6]
SSE= np.sum((y-y_pred)**2)
print(SSE)

mean_total = np.mean(dataset.values)
SST= np.sum((dataset.values-mean_total)**2)
print(SST)

r_square = 1-(SSR / SST)
r_square

print(r_square)
print(bias_score)
print(variance_score)









