import scipy as sp
from numpy.linalg import det, inv
from functools import reduce

def estimate_gaussian(X):
  mu = sp.mean(sp.asarray(X), 0)
  sigma2 = sp.mean(sp.asarray(X - mu) ** 2, 0)
  return [mu, sigma2]

def multivariate_gaussian(X, mu, sigma2):
  n = mu.size
  if (sigma2.ndim == 1) or (sigma2.shape[1] == 1):
    sigma2 = sp.diag(sigma2)
  p = (1 / ((2 * sp.pi) ** (n / 2) * det(sigma2) ** 0.5)) * \
        sp.exp(-0.5 * sp.sum(sp.asarray((X - mu) * sp.matrix(inv(sigma2))) * sp.asarray(X - mu), 1))
  return p

def select_threshold(Y_val, p_val):
  def accumulator(acc, curr):
    predictions = sp.double(p_val < curr)
    tp = sp.sum(sp.logical_and(predictions == 1, sp.asarray(Y_val == 1).ravel()))
    fp = sp.sum(sp.logical_and(predictions == 1, sp.asarray(Y_val == 0).ravel()))
    fn = sp.sum(sp.logical_and(predictions == 0, sp.asarray(Y_val == 1).ravel()))
    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    F1 = 2 * prec * rec / (prec + rec)
    return { 'epsilon': curr, 'F1': F1 } if F1 > acc['F1'] else acc

  epsilons = sp.linspace(sp.amin(p_val), sp.amax(p_val), 1000).tolist()
  return reduce(accumulator, epsilons, { 'epsilon': 0, 'F1': 0 })
