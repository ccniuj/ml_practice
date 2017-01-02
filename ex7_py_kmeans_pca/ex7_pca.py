import scipy as sp
from scipy.io import loadmat
import matplotlib.pyplot as plt
from matplotlib import colors
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from funcs import feature_normalize, project_data, recover_data, display_data, find_closest_centroids

# Load data from source 1
data = loadmat('ex7data1.mat')
X = sp.matrix(data['X'])

# Run PCA
X_norm, mu, sigma = feature_normalize(X)
pca = PCA()
pca.fit(X_norm)
U = pca.components_
S = pca.explained_variance_

# Plot eigenvector
p1 = 1.5 * S[0] * U[:, 0].T
p2 = 1.5 * S[1] * U[:, 1].T
plt.figure(4)
plt.axis('equal')
plt.plot(X[:, 0], X[:, 1], 'bo')
plt.plot([mu[0, 0], mu[0, 0] + p1[0]], [mu[0, 1], mu[0, 1] + p1[1]], '-k', linewidth=2)
plt.plot([mu[0, 0], mu[0, 0] + p2[0]], [mu[0, 1], mu[0, 1] + p2[1]], '-k', linewidth=2)
plt.savefig('4.png')

# Project data
K = 1
Z = project_data(X_norm, U, K)

# Recover
X_rec = recover_data(Z, U, K)

# Plot
plt.figure(5)
plt.axis('equal')
plt.plot(X_norm[:, 0], X_norm[:, 1], 'bo')
plt.plot(X_rec[:, 0], X_rec[:, 1], 'ro')
for i in range(0, X_rec.shape[0]):
  plt.plot([X_norm[i, 0], X_rec[i, 0]], [X_norm[i, 1], X_rec[i, 1]], '--k', linewidth=1)
plt.savefig('5.png')


# Load data from face dataset
data = loadmat('ex7faces.mat')
X = sp.matrix(data['X'])

# Display data
display_data(X[0:100], save=True, file_name='6.png')

# Run PCA on face dataset
X_norm, mu, sigma = feature_normalize(X)
pca = PCA()
pca.fit(X_norm)
U = pca.components_
S = pca.explained_variance_

# Display data
display_data(U[0:36, :], save=True, file_name='7.png')

# Project data
K = 100
Z = project_data(X_norm, U, K)

# Recover
X_rec = recover_data(Z, U, K)

# Display data
display_data(X_rec[0:100], save=True, file_name='8.png')


# Load image
data = plt.imread('bird_small.png')
X = data.reshape(data.shape[0] * data.shape[1], 3)
K = 16
max_iters = 10

# Run scikit-learn K-means
model = KMeans(n_clusters=K, random_state=0).fit(X)
centroids = model.cluster_centers_
idx = find_closest_centroids(X, centroids)

# Sample 1000 random indices
sel = sp.random.randint(0, X.shape[0], size=1000)

palette = sp.array(list(colors.cnames.keys()))[0:K]
c = palette[idx[sel]].ravel()

# Plot 3D
plt.figure(9)
plt.gca(projection='3d')
plt.scatter(X[sel, 0], X[sel, 1], X[sel, 2], c=c)
plt.savefig('9.png')

# Run PCA
X_norm, mu, sigma = feature_normalize(X)
pca = PCA()
pca.fit(X_norm)
U = pca.components_
S = pca.explained_variance_

# Project data
Z = project_data(X_norm, U, 2)

# Plot 2D
plt.figure(10)
plt.scatter(Z[sel, 0], Z[sel, 1], c=c)
plt.savefig('10.png')
