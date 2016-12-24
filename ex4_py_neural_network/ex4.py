import scipy as sp
from scipy.io import loadmat
from scipy.optimize import fmin_cg
from funcs import display_data, nn_cost_function, nn_gradients, rand_initialize_weights, check_nn_gradients, forward_prop

input_layer_size = 400
hidden_layer_size = 25
num_labels = 10

# Load data
data = loadmat('ex4data1.mat')
X = sp.matrix(data['X'])
Y = sp.matrix(data['y'])
m = X.shape[0]

rand_indices = sp.random.randint(0, m, size=100)
sel = X[rand_indices, :]

display_data(sel, save=True, file_name='1.png')

# Load calculated weights
weights = loadmat('ex4weights.mat')
theta_1 = sp.matrix(weights['Theta1'])
theta_2 = sp.matrix(weights['Theta2'])

# Unroll parameters
nn_params = sp.concatenate([sp.ravel(theta_1), sp.ravel(theta_2)])

# Initialize theta randomly
init_theta_1 = rand_initialize_weights(input_layer_size, hidden_layer_size)
init_theta_2 = rand_initialize_weights(hidden_layer_size, num_labels)

# Unroll parameters
init_nn_params = sp.concatenate([sp.ravel(init_theta_1), sp.ravel(init_theta_2)])

# Check gradients
_lambda_check = 3
diff = check_nn_gradients(_lambda_check)
print("Perform gradient checking...\nRelative difference:\n{diff}\n".format(**locals()))

# Train NN
_lambda = 1

# Debug
c = 0
def debug(_):
  global c
  c = c + 1
  print("Iteration #{c}".format(**globals()))

print('Training...')
nn_params = fmin_cg(
              nn_cost_function(input_layer_size, hidden_layer_size, num_labels, X, Y, _lambda),
              init_nn_params,
              fprime=nn_gradients(input_layer_size, hidden_layer_size, num_labels, X, Y, _lambda),
              maxiter=50,
              callback=debug
            )

# Reshape
theta_1 = nn_params[0:(hidden_layer_size*(input_layer_size+1))]. \
            reshape([hidden_layer_size, input_layer_size+1])
theta_2 = nn_params[(hidden_layer_size*(input_layer_size+1)):]. \
            reshape([num_labels, hidden_layer_size+1])

# Display data
display_data(theta_1[:, 1:], save=True, file_name='2.png')

# Estimate performance
a = forward_prop(X)(theta_1, theta_2)['a'][-1]
prep = sp.mod(sp.argmax(a, axis=1)+1, 10)
r = sp.around(sp.mean(sp.double(prep == (Y%10))) * 100, 1)
print("Train Accuracy (Neural Network): {r}%".format(**locals()))
