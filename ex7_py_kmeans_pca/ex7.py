import scipy as sp
from scipy.io import loadmat
import matplotlib.pyplot as plt
from funcs import run_k_means, k_means_init_centroids, find_closest_centroids
from sklearn.cluster import KMeans

# Load data from source 2
data = loadmat('ex7data2.mat')
X = sp.matrix(data['X'])

# Run K-means
K = 3
max_iters = 10
init_centroids = sp.matrix([[3, 3], [6, 2], [8, 5]])
centroids = run_k_means(X, init_centroids, max_iters, plot_progress=True)

# Load image
data = plt.imread('bird_small.png')
X = data.reshape(data.shape[0] * data.shape[1], 3)
K = 16
max_iters = 10

# Run implemented K-means
init_centroids = k_means_init_centroids(X, K)
centroids = run_k_means(X, init_centroids, max_iters)
# Recover
idx = find_closest_centroids(X, centroids)
img = sp.asarray(centroids[sp.asarray(idx).ravel(), :]).reshape(data.shape[0], data.shape[1], 3)
plt.figure(2)
plt.imshow(img)
plt.savefig('2.png')

# Run scikit-learn K-means, which is much faster
model = KMeans(n_clusters=K, random_state=0).fit(X)
centroids = model.cluster_centers_
# Recover
idx = find_closest_centroids(X, centroids)
img = sp.asarray(centroids[sp.asarray(idx).ravel(), :]).reshape(data.shape[0], data.shape[1], 3)
plt.figure(3)
plt.imshow(img)
plt.savefig('3.png')
