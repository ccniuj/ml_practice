import scipy as sp
from scipy.io import loadmat
import matplotlib.pyplot as plt
from funcs import train, learning_curve, validation_curve, poly_features, feature_normalize

# Load data
data = loadmat('ex5data1.mat')
X = sp.matrix(data['X'])
Y = sp.matrix(data['y'])
X_val = sp.matrix(data['Xval'])
Y_val = sp.matrix(data['yval'])
X_test = sp.matrix(data['Xtest'])
Y_test = sp.matrix(data['ytest'])

# Initialze
m = X.shape[0]
m_val = X_val.shape[0]
m_test = X_test.shape[0]
X_a = sp.hstack((sp.ones((m, 1)), X))
X_val_a = sp.hstack((sp.ones((m_val, 1)), X_val))

# Map features
p = 8
X_n, mu, sigma = feature_normalize(poly_features(X, p))
X_poly = sp.hstack((sp.ones((m, 1)), X_n))
X_poly_val = sp.hstack((sp.ones((m_val, 1)), feature_normalize(poly_features(X_val, p), mu, sigma)[0]))
X_poly_test = sp.hstack((sp.ones((m_test, 1)), feature_normalize(poly_features(X_val, p), mu, sigma)[0]))

# Train
_lambda = 0.3
theta = train(X_poly, Y, _lambda)

# Initialze variable for plot
X_plot = sp.matrix(sp.arange(sp.amin(X)-15, sp.amax(X)+25, 0.05)).T
m_plot = X_plot.shape[0]
X_poly_plot = sp.hstack((sp.ones((m_plot, 1)), feature_normalize(poly_features(X_plot, p), mu, sigma)[0]))

plt.figure(1)
plt.plot(X, Y, 'rx', markersize=10, linewidth=1.5)
plt.plot(X_plot, X_poly_plot * theta, '--', linewidth=2)
plt.xlabel('Change in water level (x)')
plt.ylabel('Water flowing out of the dam (y)')
plt.savefig('1.png')

# Plot learning curve
error_train, error_val = learning_curve(X_poly, Y, X_poly_val, Y_val, _lambda)

plt.figure(2)
plt.plot(range(1, m+1), error_train, label="Train")
plt.plot(range(1, m+1), error_val, label="Cross Validation")
plt.xlabel('Number of training examples')
plt.ylabel('Error')
plt.legend(loc='upper right')
plt.savefig('2.png')

# Model selection
lambda_vec, error_train, error_val = validation_curve(X_poly, Y, X_poly_val, Y_val)

# Plot model selection curve
plt.figure(3)
plt.plot(lambda_vec, error_train, label="Train")
plt.plot(lambda_vec, error_val, label="Cross Validation")
plt.xlabel('lambda')
plt.ylabel('Error')
plt.legend(loc='upper right')
plt.savefig('3.png')
