import scipy as sp
import matplotlib
import matplotlib.pyplot as plt
from functools import reduce

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

def forward_prop(X):
  def forward(acc, theta):
    X = acc['a'][-1]
    m = X.shape[0]

    # Add bias
    X = sp.hstack((sp.ones((m, 1)), X))
    # Update 'a'
    acc['a'][-1] = X
    # Compute
    z = X * theta.T
    a = sigmoid(z)
    # Accumulate
    acc['z'].append(z)
    acc['a'].append(a)

    return acc

  init = { 'a': [X], 'z': [] }
  return lambda *thetas: reduce(forward, thetas, init)

def sigmoid(z):
  return 1 / (1 + sp.exp(-z))

def sigmoid_gradient(z):
  return sp.multiply(sigmoid(z), (1- sigmoid(z)))

def nn_cost_function(input_layer_size, hidden_layer_size, num_labels, X, Y, _lambda):
  def compute(nn_params):
    m = Y.shape[0]

    # Reshape nn_params back into the parameters theta_1 and theta_2
    theta_1 = nn_params[0:(hidden_layer_size*(input_layer_size+1))]. \
                reshape([hidden_layer_size, input_layer_size+1])
    theta_2 = nn_params[(hidden_layer_size*(input_layer_size+1)):]. \
                reshape([num_labels, hidden_layer_size+1])

    theta_1_reg = sp.copy(theta_1)
    theta_1_reg[:, 0] = 0
    theta_2_reg = sp.copy(theta_2)
    theta_2_reg[:, 0] = 0

    # Forward propagation
    f = forward_prop(X)(theta_1, theta_2)
    a = f['a']
    a_3 = a[2]

    # Transform Y
    b = sp.matrix(sp.apply_along_axis(lambda n: sp.int_(sp.array(range(1, num_labels+1)) == n), 1, Y))

    J = 0

    for i in range(0, m):
      J = J + (1 / m) * (-b[i, :] * sp.log(a_3[i, :].T) - (1 - b[i, :]) * sp.log(1 - a_3[i, :].T))[0, 0]

    # Regularize
    J = J + (_lambda / (2 * m)) * (sp.sum(sp.power(theta_1_reg, 2)) + sp.sum(sp.power(theta_2_reg, 2))).real

    return J

  return compute

def nn_gradients(input_layer_size, hidden_layer_size, num_labels, X, Y, _lambda):
  def compute(nn_params):
    m = Y.shape[0]

    # Reshape nn_params back into the parameters theta_1 and theta_2
    theta_1 = nn_params[0:(hidden_layer_size*(input_layer_size+1))]. \
                reshape([hidden_layer_size, input_layer_size+1])
    theta_2 = nn_params[(hidden_layer_size*(input_layer_size+1)):]. \
                reshape([num_labels, hidden_layer_size+1])

    theta_1_reg = sp.copy(theta_1)
    theta_1_reg[:, 0] = 0
    theta_2_reg = sp.copy(theta_2)
    theta_2_reg[:, 0] = 0

    # Forward propagation
    f = forward_prop(X)(theta_1, theta_2)

    # Initialize variables for back propagation
    a = f['a']

    # Add bias
    a_1 = a[0]
    a_2 = a[1]
    a_3 = a[2]

    z = f['z']
    z_2 = z[0]
    z_3 = z[1]

    # Transform Y
    b = sp.matrix(sp.apply_along_axis(lambda n: sp.int_(sp.array(range(1, num_labels+1)) == n), 1, Y))

    DEL_1 = sp.matrix(sp.zeros((hidden_layer_size, input_layer_size+1)))
    DEL_2 = sp.matrix(sp.zeros((num_labels, hidden_layer_size+1)))

    for i in range(0, m):
      del_3 = a_3[i, :].T - b[i, :].T
      del_2 = sp.multiply(theta_2[:, 1:].T * del_3, sigmoid_gradient(z_2[i, :].T))

      DEL_2 = DEL_2 + del_3 * a_2[i, :]
      DEL_1 = DEL_1 + del_2 * a_1[i, :]

    # Regularize
    theta_1_grad = DEL_1 / m + (_lambda / m) * theta_1_reg
    theta_2_grad = DEL_2 / m + (_lambda / m) * theta_2_reg
    grad = sp.concatenate([sp.ravel(theta_1_grad), sp.ravel(theta_2_grad)])

    return grad

  return compute

def rand_initialize_weights(L_in, L_out):
  epsilon_init = 0.12
  return sp.random.rand(L_out, L_in+1) * 2 * epsilon_init - epsilon_init

def debug_initialize_weights(L_in, L_out):
  return sp.matrix(sp.sin(sp.array(range(1, L_out*(L_in+1)+1))). \
                      reshape((L_out, L_in+1)) / 10)

def check_nn_gradients(_lambda=0):
  input_layer_size = 3
  hidden_layer_size = 5
  num_labels = 3
  m = 5
  theta_1 = debug_initialize_weights(input_layer_size, hidden_layer_size)
  theta_2 = debug_initialize_weights(hidden_layer_size, num_labels)
  X  = debug_initialize_weights(input_layer_size-1, m)
  Y  = 1 + sp.matrix(sp.mod(range(0, m), num_labels)).T
  nn_params = sp.concatenate([sp.ravel(theta_1), sp.ravel(theta_2)])

  D = nn_gradients(input_layer_size, hidden_layer_size, num_labels, X, Y, _lambda)(nn_params)
  N = compute_numerical_gradient(nn_cost_function(input_layer_size, hidden_layer_size, num_labels, X, Y, _lambda), nn_params)
  diff = sp.linalg.norm(N-D) / sp.linalg.norm(N+D)
  return diff

def compute_numerical_gradient(cost_func, theta):
  e = sp.power(10, -4)
  numgrad = sp.zeros(theta.shape)
  perturb = sp.zeros(theta.shape)
  for i in range(0, theta.size):
    perturb[i] = e
    numgrad[i] = (cost_func(theta+perturb) - cost_func(theta-perturb)) / (2 * e)
    perturb[i] = 0
  return numgrad
