import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv(r"E:\DATASCIENCE WITH GEN AI & AGENTIC AI\emp_sal.csv")

x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x,y)

# LInear Regression Visualization
plt.scatter(x, y, color= "red")
plt.plot(x,lin_reg.predict(x),color = "blue")
plt.title("Linear Regeression graph")
plt.xlabel("position level")
plt.ylabel("salary")
plt.show()


lin_model_pred= lin_reg.predict([[6.5]])
lin_model_pred

# Polynomial Model degree 2
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures()
x_poly = poly_reg.fit_transform(x)

poly_reg.fit(x_poly,y)

lin_reg_2 = LinearRegression()
lin_reg_2.fit(x_poly,y)

print(lin_reg)
print(poly_reg)
print(lin_reg_2)


plt.scatter(x,y,color='red')
plt.plot(x,lin_reg_2.predict(poly_reg.fit_transform(x)),color='blue')
plt.title('truth or bluff (polynomial regression)')
plt.xlabel("position level")
plt.ylabel("salary")
plt.show()

poly_model_pred= lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))
poly_model_pred


# Polynomial Model degree 3
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(5)
x_poly = poly_reg.fit_transform(x)

poly_reg.fit(x_poly,y)

lin_reg_3 = LinearRegression()
lin_reg_3.fit(x_poly,y)

print(lin_reg)
print(poly_reg)
print(lin_reg_3)


plt.scatter(x,y,color='red')
plt.plot(x,lin_reg_3.predict(poly_reg.fit_transform(x)),color='blue')
plt.title('truth or bluff (polynomial regression)')
plt.xlabel("position level")
plt.ylabel("salary")
plt.show()

poly_model_pred= lin_reg_3.predict(poly_reg.fit_transform([[6.5]]))
poly_model_pred


# svr model prediction

from sklearn.svm import SVR
svr_regressor = SVR(kernel='sigmoid',degree=3,gamma='scale')
svr_regressor.fit(x,y)
svr_model_pred=svr_regressor.predict([[6.5]])
print(svr_model_pred)


from sklearn.svm import SVR
svr_regressor = SVR(kernel='poly',degree=4,gamma='scale')
svr_regressor.fit(x,y)
svr_model_pred=svr_regressor.predict([[6.5]])
print(svr_model_pred)


from sklearn.svm import SVR
svr_regressor = SVR(kernel='poly',degree=4,gamma='auto')
svr_regressor.fit(x,y)
svr_model_pred=svr_regressor.predict([[6.5]])
print(svr_model_pred)


from sklearn.svm import SVR
svr_regressor = SVR(kernel='poly',degree=5,gamma='auto')
svr_regressor.fit(x,y)
svr_model_pred=svr_regressor.predict([[6.5]])
print(svr_model_pred)



# KNN MODEL


from sklearn.neighbors import KNeighborsRegressor
knn_regressor = KNeighborsRegressor()
knn_regressor.fit(x,y)
knn_model_pred=knn_regressor.predict([[6.5]])
knn_model_pred



from sklearn.neighbors import KNeighborsRegressor
knn_regressor = KNeighborsRegressor(n_neighbors=4,weights='distance',p=2)
knn_regressor.fit(x,y)
knn_model_pred=knn_regressor.predict([[6.5]])
knn_model_pred



from sklearn.neighbors import KNeighborsRegressor
knn_regressor = KNeighborsRegressor(n_neighbors=4,weights='distance',p=1)
knn_regressor.fit(x,y)
knn_model_pred=knn_regressor.predict([[6.5]])
knn_model_pred



from sklearn.neighbors import KNeighborsRegressor
knn_regressor = KNeighborsRegressor(n_neighbors=3,weights='distance',p=1)
knn_regressor.fit(x,y)
knn_model_pred=knn_regressor.predict([[6.5]])
knn_model_pred


from sklearn.neighbors import KNeighborsRegressor
knn_regressor = KNeighborsRegressor(n_neighbors=2,weights='distance',p=1)
knn_regressor.fit(x,y)
knn_model_pred=knn_regressor.predict([[6.5]])
knn_model_pred


from sklearn.neighbors import KNeighborsRegressor
knn_regressor = KNeighborsRegressor(n_neighbors=2)
knn_regressor.fit(x,y)
knn_model_pred=knn_regressor.predict([[6.5]])
knn_model_pred


# TREE ALGORITHM

from sklearn.tree import DecisionTreeRegressor
dt_regressor = DecisionTreeRegressor()
dt_regressor.fit(x, y)

dt_model_pred= dt_regressor.predict([[6.5]])
dt_model_pred

# RANDOM FOREST

from sklearn.ensemble import RandomForestRegressor
rf_regressor = RandomForestRegressor()
rf_regressor.fit(x, y)

rf_model_pred= rf_regressor.predict([[6.5]])
rf_model_pred


from sklearn.ensemble import RandomForestRegressor
rf_regressor = RandomForestRegressor(random_state=0,n_estimators=8)
rf_regressor.fit(x, y)

rf_model_pred= rf_regressor.predict([[6.5]])
rf_model_pred

















