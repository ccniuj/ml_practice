import scipy as sp
import matplotlib
import matplotlib.pyplot as plt
from scipy.optimize import fmin_bfgs
from functools import reduce

def display_data(X, width=None, save=False):
  m, n = X.shape
  width = sp.int_(width or sp.around(sp.sqrt(n)))
  height = sp.int_(n / width)
  display_rows = sp.int_(sp.floor(sp.sqrt(m)))
  display_cols = sp.int_(sp.ceil(m / display_rows))

  def rightward(acc, curr):
    return sp.hstack([acc, curr])

  def downward(acc, curr):
    return sp.vstack([acc, curr])

  def merge(func, init):
    return lambda arr: reduce(func, arr, init)

  init_rightward = sp.matrix([]).reshape([height, 0])
  init_downward = sp.matrix([]).reshape([0, width * display_cols])

  img_list = [X[i].reshape([height, width]).T for i in range(0, m)]
  img_list_split = [img_list[i:i+display_cols] for i in range(0, len(img_list), display_cols)]
  img = merge(downward, init_downward)(map(merge(rightward, init_rightward), img_list_split))

  plt.figure(1)
  plt.imshow(img, cmap='gray')
  plt.tick_params(labelbottom='off', labelleft='off')
  if save:
    plt.savefig('1.png')
  else:
    plt.show()

  return None

def one_vs_all(X, Y, num_labels, _lambda, cost_func, grad_func):
  m, n = X.shape
  X = sp.hstack((sp.ones((m, 1)), X))
  all_theta = sp.zeros([num_labels, n+1])

  for c in range(1, num_labels+1):
    init_theta = sp.ones(X.shape[1])
    all_theta[c%num_labels, :] = fmin_bfgs(cost_func, init_theta, fprime=grad_func, args=(X, sp.int_(Y==c), 0), maxiter=100)

  return all_theta

def predict_one_vs_all(all_theta, X):
  m = X.shape[0]
  X = sp.hstack((sp.ones((m, 1)), X))
  return sp.argmax(sigmoid(X * all_theta.T), axis=1)

def forward_prop(X):
  def forward(X, theta):
    m = X.shape[0]
    X = sp.hstack((sp.ones((m, 1)), X))
    return sigmoid(X * theta.T)

  return lambda *thetas: sp.mod(sp.argmax(reduce(forward, thetas, X), axis=1)+1, 10)

def sigmoid(z):
  return 1 / (1 + sp.exp(-z))

def cost_function_reg(theta, X, Y, _lambda):
  theta = sp.matrix(theta).T
  theta_reg = sp.copy(theta)
  theta_reg[0] = 0
  m = Y.shape[0]

  J = (1 / m) * (-Y.T * sp.log(sigmoid(X * theta)) - ((1 - Y).T * sp.log(1 - sigmoid(X * theta)))) \
        + (_lambda / (2 * m)) * (theta_reg.T * theta_reg)
  return J[0, 0]

def gradients_reg(theta, X, Y, _lambda):
  theta = sp.matrix(theta).T
  theta_reg = sp.copy(theta)
  theta_reg[0] = 0
  m = Y.shape[0]

  grad = ((1 / m) * (X.T * (sigmoid(X * theta) - Y) + _lambda * theta_reg)).T
  grad = sp.squeeze(sp.asarray(grad))
  return grad
