import scipy as sp
from scipy.io import loadmat
from funcs import display_data, one_vs_all, predict_one_vs_all, forward_prop, cost_function_reg, gradients_reg

input_layer_size = 400
num_labels = 10

# Load data
data = loadmat('ex3data1.mat')
X = sp.matrix(data['X'])
Y = sp.matrix(data['y'])
m = X.shape[0]

rand_indices = sp.random.randint(0, m, size=100)
sel = X[rand_indices, :]

display_data(sel, save=True)

# Logistic regression
_lambda = 0.1
all_theta = one_vs_all(X, Y, num_labels, _lambda, cost_function_reg, gradients_reg)
p = predict_one_vs_all(all_theta, X)
r = sp.around(sp.mean(sp.double(p == (Y%10))) * 100, 1)

print("Train Accuracy (Logistic Regression): {r}%".format(**locals()))

# Neural network
hidden_layer_size = 25

# Load calculated weights
weights = loadmat('ex3weights.mat')
theta_1 = sp.matrix(weights['Theta1'])
theta_2 = sp.matrix(weights['Theta2'])

prep = forward_prop(X)(theta_1, theta_2)
r = sp.around(sp.mean(sp.double(prep == (Y%10))) * 100, 1)

print("Train Accuracy (Neural Network): {r}%".format(**locals()))

for i in rand_indices.tolist():
  prep = forward_prop(X[i])(theta_1, theta_2)[0, 0]
  print("Predicted digit: {prep}".format(**locals()))
  display_data(X[i])
