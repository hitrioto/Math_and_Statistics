import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Generate synthetic data
np.random.seed(0)
x = np.linspace(-3, 3, 100)
y = 1 + 2 * x - x**2 + np.random.randn(*x.shape) * 2

# Reshape x for sklearn
X = x[:, np.newaxis]

# Transform X to include polynomial terms up to degree 2
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

# Perform linear regression on the transformed X
model = LinearRegression().fit(X_poly, y)

# Predict y values using the model
y_pred = model.predict(X_poly)

# Plotting
plt.figure(figsize=(8, 6))
plt.scatter(x, y, label='Data Points')
plt.plot(x, y_pred, color='red', label='Polynomial Regression Fit')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Polynomial Regression Example')
plt.legend()
plt.show()
