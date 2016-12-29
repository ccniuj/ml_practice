import scipy as sp
from scipy.io import loadmat
import matplotlib.pyplot as plt
from sklearn import svm

# Load data from source 1
data = loadmat('ex6data1.mat')
X = sp.matrix(data['X'])
Y = sp.matrix(data['y'])

# Train SVM
model1 = svm.LinearSVC()
model1.fit(X, sp.array(Y).ravel())

# Visualize data and decision boundary
w = model1.coef_[0]
b = model1.intercept_

xp = sp.linspace(sp.amin(X[:, 0]), sp.amax(X[:, 0]), 100)
yp = -(w[0] * xp + b) / w[1]

plt.figure(1)
plt.xlabel('x1')
plt.ylabel('x2')

pos = sp.where(Y == 1)[0]
neg = sp.where(Y == 0)[0]

plt.plot(X[pos, 0], X[pos, 1], 'k+', linewidth=1, markersize=7)
plt.plot(X[neg, 0], X[neg, 1], 'ko', markerfacecolor='y', markersize=7)
plt.plot(xp, yp, '-b')

plt.savefig('1.png')

# Load data from source 2
data = loadmat('ex6data2.mat')
X = sp.matrix(data['X'])
Y = sp.matrix(data['y'])

# Train SVM
C = 1
sigma = 0.1
gamma = 1 / (2 * sigma ** 2)

model2 = svm.SVC(gamma=gamma)
model2.fit(X, sp.array(Y).ravel())

# Plot data and decision boundary
plt.figure(2)
plt.xlabel('x1')
plt.ylabel('x2')

pos = sp.where(Y == 1)[0]
neg = sp.where(Y == 0)[0]

plt.plot(X[pos, 0], X[pos, 1], 'k+', linewidth=1, markersize=7)
plt.plot(X[neg, 0], X[neg, 1], 'ko', markerfacecolor='y', markersize=7)

n = 100
x1p = sp.linspace(sp.amin(X[:, 0]), sp.amax(X[:, 0]), n)
x2p = sp.linspace(sp.amin(X[:, 1]), sp.amax(X[:, 1]), n)
X1, X2 = sp.meshgrid(x1p, x2p)

vals = sp.matrix([model2.predict(sp.vstack((X1[i, :], X2[i, :])).T) for i in range(0, n)]).reshape((n, n))
plt.contour(X1, X2, vals, [0], linecolor='b')
plt.savefig('2.png')


# Load data from source 3
data = loadmat('ex6data3.mat')
X = sp.matrix(data['X'])
Y = sp.matrix(data['y'])
X_val = sp.matrix(data['Xval'])
Y_val = sp.matrix(data['yval'])

# Model selection
C_vec = [0.1, 0.3, 1, 3, 10]
sigma_vec = [0.01, 0.03, 0.1, 0.3, 1]
comb = [{ 'C': C_vec[i], 'sigma': sigma_vec[j] }
         for i in range(0, len(C_vec))
         for j in range(0, len(sigma_vec))]

C = None
sigma = None
error = sp.inf

for i in range(0, len(comb)):
  # Train
  C_t = comb[i]['C']
  sigma_t = comb[i]['sigma']
  gamma_t = 1 / (2 * sigma_t ** 2)
  model_t = svm.SVC(C=C_t, gamma=gamma_t)
  model_t.fit(X, sp.array(Y).ravel())
  # Predict
  p = sp.matrix(model_t.predict(X_val)).T
  error_t = sp.mean(sp.double(p != Y_val))
  if error_t < error:
    error = error_t
    C = C_t
    sigma = sigma_t

# Train
gamma = 1 / (2 * sigma ** 2)

model3 = svm.SVC(gamma=gamma)
model3.fit(X, sp.array(Y).ravel())

plt.figure(3)
plt.xlabel('x1')
plt.ylabel('x2')

pos = sp.where(Y == 1)[0]
neg = sp.where(Y == 0)[0]

plt.plot(X[pos, 0], X[pos, 1], 'k+', linewidth=1, markersize=7)
plt.plot(X[neg, 0], X[neg, 1], 'ko', markerfacecolor='y', markersize=7)

n = 100
x1p = sp.linspace(sp.amin(X[:, 0]), sp.amax(X[:, 0]), n)
x2p = sp.linspace(sp.amin(X[:, 1]), sp.amax(X[:, 1]), n)
X1, X2 = sp.meshgrid(x1p, x2p)
vals = sp.matrix([model3.predict(sp.vstack((X1[i, :], X2[i, :])).T) for i in range(0, n)]).reshape((n, n))
plt.contour(X1, X2, vals, [0], linecolor='b')

plt.savefig('3.png')
