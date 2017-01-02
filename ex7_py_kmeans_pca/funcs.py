import scipy as sp
import matplotlib.pyplot as plt
from matplotlib import colors
from functools import reduce
from pdb import set_trace

def find_closest_centroids(X, centroids):
  def distance_sqr(x1, x2):
    return sp.sum(sp.power(sp.array(x1) - sp.array(x2), 2))

  def find_closest_centroid_index(centroids):
    return lambda x: sp.argmin([distance_sqr(x, c) for c in centroids.tolist()])

  return sp.matrix( \
           [find_closest_centroid_index(centroids)(x) for x in X.tolist()] \
         ).T

def compute_centroids(X, idx, K):
  def compute_mean(x):
    return sp.squeeze(sp.asarray(sp.sum(x, 0))) / x.shape[0]

  return sp.matrix([compute_mean(X[sp.where(idx==k)[0]]) for k in range(0, K)])

def k_means_init_centroids(X, K):
  randidx = sp.random.randint(0, X.shape[0], size=K)
  return sp.matrix(X[randidx])

def run_k_means(X, init_centroids, max_iters, plot_progress=False):
  K = init_centroids.shape[0]
  centroids = init_centroids
  prev_centroids = centroids

  for i in range(0, max_iters):
    idx = find_closest_centroids(X, centroids)
    if plot_progress:
      plt.figure(1)
      plot_progress_k_means(X, centroids, prev_centroids, idx, K)
    prev_centroids = centroids
    centroids = compute_centroids(X, idx, K)

  if plot_progress:
    plt.savefig('1.png')

  return centroids

def plot_progress_k_means(X, centroids, previous, idx, K):
  # Plot data points
  palette = sp.array(list(colors.cnames.keys()))
  c = palette[idx].ravel()
  plt.scatter(X[:, 0], X[:, 1], c=c)

  # Plot centroids
  plt.plot(centroids[:, 0], centroids[:, 1], 'x', markeredgecolor='k', markersize=10, linewidth=3)

  # Plot trajectory
  for k in range(0, K):
    plt.plot([previous[k, 0], centroids[k, 0]], [previous[k, 1], centroids[k, 1]], 'b')

def display_data(X, width=None, save=False, file_name='1.png'):
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
    plt.savefig(file_name)
  else:
    plt.show()

  return None

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

def project_data(X, U, K):
  return sp.matrix(X) * sp.matrix(U[0:K, :].T)

def recover_data(Z, U, K):
  return sp.matrix(Z) * sp.matrix(U[0:K, :])
