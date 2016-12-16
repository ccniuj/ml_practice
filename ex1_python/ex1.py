import scipy as sp
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

print("Running warmUpExercise ...")
data = sp.matrix(sp.loadtxt("ex1data1.txt", delimiter=','))
X = data[:, 0]
Y = data[:, 1]
m = Y.shape[0]

plt.figure(1)
plt.plot(X, Y, 'ro')
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.savefig('1.png')

X = sp.hstack((sp.ones((m, 1)), X))
theta = sp.matrix(sp.zeros((2, 1)))

iterations = 1500
alpha = 0.01

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

[theta, _] = gradient_descent(X, Y, theta, alpha, iterations)

plt.plot(X[:, 1], X*theta, '-')

theta0_vals = sp.linspace(-10, 10, 100)
theta1_vals = sp.linspace(-1, 4, 100)
J_vals = sp.matrix(sp.zeros((theta0_vals.size, theta1_vals.size)))

for i, t0 in sp.ndenumerate(theta0_vals):
  for j, t1 in sp.ndenumerate(theta1_vals):
    t = sp.matrix([[t0], [t1]])
    J_vals[i[0], j[0]] = compute_cost(X, Y, t)

J_vals = J_vals.T

fig = plt.figure(2)
ax = fig.gca(projection='3d')
X, Y = sp.meshgrid(theta0_vals, theta1_vals)
Z = J_vals
surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
plt.savefig('2.png')

plt.figure(3)
matplotlib.rcParams['xtick.direction'] = 'out'
matplotlib.rcParams['ytick.direction'] = 'out'

X, Y = sp.meshgrid(theta0_vals, theta1_vals)
Z = J_vals

CS = plt.contour(X, Y, Z, sp.logspace(-2, 3, 20))
plt.clabel(CS, inline=1, fontsize=10)
plt.plot(theta[0, 0], theta[1, 0], 'ro')
plt.savefig('3.png')
plt.show()
