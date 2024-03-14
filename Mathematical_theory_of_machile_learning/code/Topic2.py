# example of a logistic regression
#%%
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
import numpy as np

# Generate a toy 2D dataset
X, y = make_classification(n_samples=100, n_features=2, n_redundant=0, n_clusters_per_class=1, random_state=42)

# Fit logistic regression model
model = LogisticRegression()
model.fit(X, y)

# Plot the data points
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolors='k', label='Data Points')

# Create grid to plot decision boundaries
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))

# Predict over the grid
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot the decision boundary 

plt.contour(xx, yy, Z, colors='red', levels=[0.5], linestyles='dashed', linewidths=2)  # Adjusted for visibility
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Logistic Regression Decision Boundary')
plt.legend()
plt.show()
#%%
# 3D plot for the dependent variable

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit as logistic_function

# Generate a meshgrid for the feature space
x = np.linspace(-10, 10, 100)
y = np.linspace(-10, 10, 100)
X, Y = np.meshgrid(x, y)
Z = logistic_function(0.5 * X - 0.4 * Y - 0.1)  # Example linear combination

# Create a 3D plot
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Plot the surface
surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')

# Labels and title
ax.set_xlabel('Feature 1 (X)')
ax.set_ylabel('Feature 2 (Y)')
ax.set_zlabel('Probability (Z)')
ax.set_title('3D Visualization of Logistic Regression Probability')

# Add a color bar
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()

# %%
