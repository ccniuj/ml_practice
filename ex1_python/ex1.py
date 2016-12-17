import scipy as sp
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

# Load data from data source 1
data = sp.matrix(sp.loadtxt("ex1data1.txt", delimiter=','))
X_raw = data[:, 0]
Y = data[:, 1]
m = Y.shape[0]

# Implement cost function, gradient descent, feature normalize and normal equation
def compute_cost(X, y, theta):
  m = y.shape[0]
  J = 1 / (2 * m) * (X * theta - y).T * (X * theta - y)
  return J

def gradient_descent(X, y, theta, alpha, num_iters):
  m = y.shape[0]
  J_history = sp.matrix(sp.zeros((num_iters, 1)))
  for i in range(num_iters):
    theta = theta - (alpha / m) * ((X * theta - y).T * X).T
    J_history[i] = compute_cost(X, y, theta)
  return [theta, J_history]

def feature_normalize(X):
  if X.shape[0] == 1:
    mu = sp.mean(X, 1)
    sigma = sp.std(X, 1)
  else:
    mu = sp.mean(X, 0)
    sigma = sp.std(X, 0)

  X_norm = (X - mu) / sigma
  return X_norm

def normal_eqn(X, Y):
  theta = (X.T * X).I * X.T * Y
  return theta

# Initialize
X = sp.hstack((sp.ones((m, 1)), X_raw))
theta = sp.matrix(sp.zeros((2, 1)))
iterations = 1500
alpha = 0.01

# Calculate theta
[theta, _] = gradient_descent(X, Y, theta, alpha, iterations)

# Plot figure 1
plt.figure(1)
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.plot(X_raw, Y, 'ro')
plt.plot(X[:, 1], X*theta, '-')
plt.savefig('1.png')

# Calculate J_vals for surface plot
theta0_vals = sp.linspace(-10, 10, 100)
theta1_vals = sp.linspace(-1, 4, 100)
J_vals = sp.matrix(sp.zeros((theta0_vals.size, theta1_vals.size)))

for i, t0 in sp.ndenumerate(theta0_vals):
  for j, t1 in sp.ndenumerate(theta1_vals):
    t = sp.matrix([[t0], [t1]])
    J_vals[i[0], j[0]] = compute_cost(X, Y, t)

J_vals = J_vals.T

# Plot figure 2 (surface plot)
fig = plt.figure(2)
ax = fig.gca(projection='3d')
X, Y = sp.meshgrid(theta0_vals, theta1_vals)
Z = J_vals
surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
plt.savefig('2.png')

# Plot figure 3 (contour plot)
plt.figure(3)
matplotlib.rcParams['xtick.direction'] = 'out'
matplotlib.rcParams['ytick.direction'] = 'out'

X, Y = sp.meshgrid(theta0_vals, theta1_vals)
Z = J_vals

CS = plt.contour(X, Y, Z, sp.logspace(-2, 3, 20))
plt.clabel(CS, inline=1, fontsize=10)
plt.plot(theta[0, 0], theta[1, 0], 'ro')
plt.savefig('3.png')

# Load data from data source 2
data = sp.matrix(sp.loadtxt("ex1data2.txt", delimiter=','))
X = data[:, [0, 1]]
Y = data[:, 2]
m = Y.shape[0]

# Initialize
X = feature_normalize(X)
X = sp.hstack((sp.ones((m, 1)), X))
theta = sp.matrix(sp.zeros((3, 1)))
num_iters = 400
alpha = 0.03

# Calculate theta
[theta, J_history] = gradient_descent(X, Y, theta, alpha, num_iters)

# Plot figure 4
plt.figure(4)
plt.plot(range(0, J_history.size), J_history, '-b', linewidth=2.0);
plt.savefig('4.png')

# Predict
x_i = sp.matrix([1650, 3])
x_n = feature_normalize(x_i) # Do feature scaling
x = sp.hstack((sp.matrix([1]), x_n))
price = (x * theta)[0, 0]
print("Predicted price of a 1650 sq-ft, 3 br house (using gradient descent) to be {price}".format(**locals()))

# Load data from data source 2
data = sp.matrix(sp.loadtxt("ex1data2.txt", delimiter=','))
X = data[:, [0, 1]]
Y = data[:, 2]
m = Y.shape[0]

X = sp.hstack((sp.ones((m, 1)), X))

# Calculate thaeta by normal equation
theta = normal_eqn(X, Y)

# Predict
x_i = sp.matrix([1650, 3])
x = sp.hstack((sp.matrix([1]), x_i))
price = (x * theta)[0, 0]
print("Predicted price of a 1650 sq-ft, 3 br house (using normal equations) to be {price}".format(**locals()))
