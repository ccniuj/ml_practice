import scipy as sp
from scipy.io import loadmat
import matplotlib.pyplot as plt
from funcs import estimate_gaussian, multivariate_gaussian, select_threshold

# Load data from source 1
data = loadmat('ex8data1.mat')
X = sp.matrix(data['X'])
X_val = sp.matrix(data['Xval'])
Y_val = sp.matrix(data['yval'])

# Compute parameters
mu, sigma2 = estimate_gaussian(X)

# Compute probability
p = multivariate_gaussian(X, mu, sigma2)

# Visualize fit
con = sp.arange(0, 35, 0.5)
X1, X2 = sp.meshgrid(con, con)
X_plot = sp.vstack((X1.ravel(), X2.ravel())).T
Z = multivariate_gaussian(X_plot, mu, sigma2).reshape(X1.shape)
steps = [10 ** i for i in range(-20, 0, 3)]

plt.figure(1)
plt.plot(X[:, 0], X[:, 1], 'bx')
plt.axis([0, 30, 0, 30])
plt.contour(X1, X2, Z, steps)

# Model selection
p_val = multivariate_gaussian(X_val, mu, sigma2)
epsilon = select_threshold(Y_val, p_val)['epsilon']

# Identify anomalies
outliers = sp.argwhere(p < epsilon).ravel()

# Visualize
plt.plot(X[outliers, 0], X[outliers, 1], 'ro', linewidth=2, markersize=10)
plt.savefig('1.png')


# Load data from source 2
data = loadmat('ex8data2.mat')
X = sp.matrix(data['X'])
X_val = sp.matrix(data['Xval'])
Y_val = sp.matrix(data['yval'])

# Compute parameters
mu, sigma2 = estimate_gaussian(X)

# Compute probability
p = multivariate_gaussian(X, mu, sigma2)

# Model selection
p_val = multivariate_gaussian(X_val, mu, sigma2)
epsilon = select_threshold(Y_val, p_val)['epsilon']

# Count outliers
count = sp.sum(p < epsilon)

print("Outliers found: {count}".format(**locals()))
