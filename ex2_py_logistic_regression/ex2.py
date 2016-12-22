import scipy as sp
import matplotlib
import matplotlib.pyplot as plt
from scipy.optimize import fmin_bfgs

# Define sigmoid, cost function and gradients
def sigmoid(z):
  return 1 / (1 + sp.exp(-z))

def cost_function(theta, X, Y):
  theta = sp.matrix(theta).T
  m = Y.shape[0]

  J = (1 / m) * (-Y.T * sp.log(sigmoid(X * theta)) - ((1 - Y).T * sp.log(1 - sigmoid(X * theta))))
  return J[0, 0]

def cost_function_reg(theta, X, Y, _lambda):
  theta = sp.matrix(theta).T
  theta_reg = sp.copy(theta)
  theta_reg[0] = 0
  m = Y.shape[0]

  J = (1 / m) * (-Y.T * sp.log(sigmoid(X * theta)) - ((1 - Y).T * sp.log(1 - sigmoid(X * theta)))) \
        + (_lambda / (2 * m)) * (theta_reg.T * theta_reg)
  return J[0, 0]

def gradients(theta, X, Y):
  theta = sp.matrix(theta).T
  m = Y.shape[0]

  grad = ((1 / m) * X.T * (sigmoid(X * theta) - Y)).T
  grad = sp.squeeze(sp.asarray(grad))
  return grad

def gradients_reg(theta, X, Y, _lambda):
  theta = sp.matrix(theta).T
  theta_reg = sp.copy(theta)
  theta_reg[0] = 0
  m = Y.shape[0]

  grad = ((1 / m) * (X.T * (sigmoid(X * theta) - Y) + _lambda * theta_reg)).T
  grad = sp.squeeze(sp.asarray(grad))
  return grad

def predict(theta, X):
  return sp.around(sigmoid(X * theta))

def map_feature(X1, X2):
  degree = 6
  out = sp.matrix([
          list(map(lambda c: c.real, sp.multiply(sp.power(X1, (i-j)), sp.power(X2, j)).flatten().tolist()))
            for i in range(0, degree+1) 
            for j in range(0, i+1)
        ]).T
  return out

# Load data from data source 1
data = sp.matrix(sp.loadtxt("ex2data1.txt", delimiter=','))
X = data[:, 0:2]
Y = data[:, 2]
m, n = X.shape

# Compute cost and gradients
# Initialize
X = sp.hstack((sp.ones((m, 1)), X))
theta = sp.zeros(n+1) # Use row vector instead of column vector for applying optimization

# Optimize using fmin_bfgs
res = fmin_bfgs(cost_function, theta, fprime=gradients, args=(X, Y))
theta = sp.matrix(res).T

# Plot fiqure 1 (decision boundary)
plt.figure(1)
plt.xlabel('x1')
plt.ylabel('x2')

pos = sp.where(Y == 1)[0]
neg = sp.where(Y == 0)[0]

plt.plot(X[pos, 1], X[pos, 2], 'k+', linewidth=2, markersize=7)
plt.plot(X[neg, 1], X[neg, 2], 'ko', markerfacecolor='y', markersize=7)

plot_x = sp.array([sp.amin(X[:, 1]) - 2, sp.amax(X[:, 1]) + 2])
plot_y = (-1 / theta[2, 0]) * (theta[0, 0] + theta[1, 0] * plot_x)
plt.plot(plot_x, plot_y)
plt.savefig('1.png')

# Estimate performance
p = predict(theta, X)
r = sp.around(sp.mean(sp.double(p == Y)) * 100, 1)

print("Train Accuracy: {r}%".format(**locals()))

# Regularize

# Load data from data source 2
data = sp.matrix(sp.loadtxt("ex2data2.txt", delimiter=','))
X = data[:, 0:2]
Y = data[:, 2]
m, n = X.shape

# Compute regularized cost and gradients
# Initialize
X = map_feature(X[:, 0], X[:, 1])
# theta = sp.zeros(X.shape[1])
theta = sp.ones(X.shape[1])
_lambda = 0.005

# Optimize using
res = fmin_bfgs(cost_function_reg, theta, fprime=gradients_reg, args=(X, Y, _lambda))
theta = sp.matrix(res).T

# Plot figure 2
plt.figure(2)
plt.xlabel('x1')
plt.ylabel('x2')

pos = sp.where(Y == 1)[0]
neg = sp.where(Y == 0)[0]

plt.plot(X[pos, 1], X[pos, 2], 'k+', linewidth=2, markersize=7)
plt.plot(X[neg, 1], X[neg, 2], 'ko', markerfacecolor='y', markersize=7)

u = sp.linspace(-1, 1.5, 50)
v = sp.linspace(-1, 1.5, 50)
z = sp.matrix(
      sp.reshape(
        [ map_feature(u[i], v[j]) * theta
           for i in range(0, len(u))
           for j in range(0, len(v)) 
        ], [50, 50])).T

plt.contour(u, v, z, 0, linewidth=2)
plt.savefig('2.png')

# Estimate performance
p = predict(theta, X)
r = sp.around(sp.mean(sp.double(p == Y)) * 100, 1)

print("Train Accuracy: {r}%".format(**locals()))
