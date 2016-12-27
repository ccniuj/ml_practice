import scipy as sp
from scipy.optimize import fmin_cg

def cost_function_reg(theta, X, y, _lambda):
  m = y.shape[0]
  theta = sp.matrix(theta).T
  theta_reg = sp.copy(theta)
  theta_reg[0] = 0
  J = 1 / (2 * m) * (X * theta - y).T * (X * theta - y) \
        + (_lambda / (2 * m)) * (theta_reg.T * theta_reg)
  return J[0, 0]

def gradients_reg(theta, X, y, _lambda):
  m = y.shape[0]
  theta = sp.matrix(theta).T
  theta_reg = sp.copy(theta)
  theta_reg[0] = 0

  grad = ((1 / m) * (X.T * (X * theta - y) + _lambda * theta_reg)).T
  grad = sp.squeeze(sp.asarray(grad))
  return grad

def train(X, Y, _lambda):
  init_theta = sp.zeros(X.shape[1])
  res = fmin_cg(cost_function_reg, init_theta, fprime=gradients_reg, args=(X, Y, _lambda), maxiter=200)
  theta = sp.matrix(res).T
  return theta

def learning_curve(X, Y, X_val, Y_val, _lambda):
  m = X.shape[0]
  error_train = sp.zeros(m)
  error_val = sp.zeros(m)

  for i in range(1, m+1):
    theta = train(X[0:i, :], Y[0:i, :], _lambda)
    error_train[i-1] = cost_function_reg(theta.T, X[0:i, :], Y[0:i, :], 0)
    error_val[i-1] = cost_function_reg(theta.T, X_val, Y_val, 0)

  return [error_train, error_val]

def validation_curve(X, Y, X_val, Y_val):
  lambda_vec = sp.array([0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3])
  m = lambda_vec.size
  error_train = sp.zeros(m)
  error_val = sp.zeros(m)

  for i in range(0, m):
    theta = train(X, Y, lambda_vec[i])
    error_train[i] = cost_function_reg(theta.T, X, Y, 0)
    error_val[i] = cost_function_reg(theta.T, X_val, Y_val, 0)

  return [lambda_vec, error_train, error_val]

def poly_features(X, p):
  X_poly = sp.zeros((X.size, p))

  for i in range(0, p):
    X_poly[:, i] = sp.squeeze(sp.array(X)) ** (i+1)

  return sp.matrix(X_poly)

def feature_normalize(X, mu=None, sigma=None):
  if (mu is None) or (sigma is None):
    if X.shape[0] == 1:
      mu = sp.mean(X, 1)
      sigma = sp.std(X, 1)
    else:
      mu = sp.mean(X, 0)
      sigma = sp.std(X, 0)

  X_norm = (X - mu) / sigma
  return [X_norm, mu, sigma]
