import operator

import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

np.random.seed(0)
x = 2 - 3*np.random.normal(0, 1, 20)
y = x - 2*(x**2) + 0.5*(x**3) + np.random.normal(-3, 3, 20)

x = x[:, np.newaxis]
y = y[:, np.newaxis]

polynomial_features = PolynomialFeatures(degree=1)
x_lin = polynomial_features.fit_transform(x)
model = LinearRegression()
model.fit(x_lin, y)
y_lin_pred = model.predict(x_lin)

polynomial_features = PolynomialFeatures(degree=2)
x_poly = polynomial_features.fit_transform(x)
model = LinearRegression()
model.fit(x_poly, y)
y_poly_pred = model.predict(x_poly)

polynomial_features = PolynomialFeatures(degree=3)
x_poly_3 = polynomial_features.fit_transform(x)
model = LinearRegression()
model.fit(x_poly_3, y)
y_poly_pred_3 = model.predict(x_poly_3)


plt.scatter(x, y, s=10, color='blue')

sort_axis = operator.itemgetter(0)
sorted_zip = sorted(zip(x, y_lin_pred), key=sort_axis)
x_poly_1, y_lin_pred = zip(*sorted_zip)
plt.plot(x_poly_1, y_lin_pred, color='lightgreen')

sort_axis = operator.itemgetter(0)
sorted_zip = sorted(zip(x, y_poly_pred), key=sort_axis)
x_poly_2, y_poly_pred = zip(*sorted_zip)
plt.plot(x_poly_2, y_poly_pred, '-.', color='lightskyblue')

sort_axis = operator.itemgetter(0)
sorted_zip = sorted(zip(x, y_poly_pred_3), key=sort_axis)
x_poly_3, y_poly_pred_3 = zip(*sorted_zip)
plt.plot(x_poly_3, y_poly_pred_3, '--', color='magenta')

plt.legend(["Degree=1", "Degree=2", "Degree=3", "Dummy Values"])

plt.show()
