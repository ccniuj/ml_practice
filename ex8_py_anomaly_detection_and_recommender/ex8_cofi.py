import scipy as sp
from scipy.io import loadmat
from scipy.optimize import fmin_cg
import matplotlib.pyplot as plt
from funcs import cofi_cost_function, cofi_gradients, check_gradients, load_movie_list, normalize_ratings

# Load movie data
data = loadmat('ex8_movies.mat')
Y = data['Y']
R = data['R']
avg_r_1 = sp.sum(Y[0]) / sp.sum(R[0])
print("Average rating for movie 1 (Toy Story): {avg_r_1} / 5".format(**locals()))

# Visualize
plt.imshow(Y)
plt.xlabel('Users')
plt.ylabel('Movies')
plt.savefig('1.png')

# Load pre-trained weights
data = loadmat('ex8_movieParams.mat')
X = data['X']
Theta = data['Theta']

# Reduce the data set size so that this runs faster
num_users = 4
num_movies = 5
num_features = 3
X = X[0:num_movies, 0:num_features]
Theta = Theta[0:num_users, 0:num_features]
Y = Y[0:num_movies, 0:num_users]
R = R[0:num_movies, 0:num_users]

J = cofi_cost_function(Y, R, num_users, num_movies, num_features, 0)(sp.hstack((X.flatten(), Theta.flatten())))
J_reg = cofi_cost_function(Y, R, num_users, num_movies, num_features, 1.5)(sp.hstack((X.flatten(), Theta.flatten())))
diff = check_gradients(_lambda=1.5)
print("Perform gradient checking...\nRelative difference:\n{diff}\n".format(**locals()))

# Load movie list
movie_list = load_movie_list()

# Initialize ratings
my_ratings = sp.zeros(movie_list.size)
my_ratings[0] = 4
my_ratings[97] = 2
my_ratings[6] = 3
my_ratings[11]= 5
my_ratings[55] = 4
my_ratings[63]= 5
my_ratings[65]= 3
my_ratings[68] = 5
my_ratings[182] = 4
my_ratings[225] = 5
my_ratings[354]= 5

# Add new ratings to dataset
data = loadmat('ex8_movies.mat')
Y = sp.hstack((sp.array([my_ratings]).T, data['Y']))
R = sp.hstack((sp.array([my_ratings != 0]).T, data['R']))
Y_norm, Y_mean = normalize_ratings(Y, R)

# Initialize
num_users = Y.shape[1]
num_movies = Y.shape[0]
num_features = 10
_lambda = 10
X = sp.random.rand(num_movies, num_features)
Theta = sp.random.rand(num_users, num_features)
init_params = sp.hstack((X.flatten(), Theta.flatten()))

# Debug
c = 0
def debug(_):
  global c
  c = c + 1
  print("Iteration #{c}".format(**globals()))

# Train
print('Training...')
params = fmin_cg(
           cofi_cost_function(Y_norm, R, num_users, num_movies, num_features, _lambda),
           init_params,
           fprime=cofi_gradients(Y_norm, R, num_users, num_movies, num_features, _lambda),
           maxiter=100,
           callback=debug
         )
X = params[0:(num_movies*num_features)]. \
      reshape([num_movies, num_features])
Theta = params[(num_movies*num_features):]. \
          reshape([num_users, num_features])

# Predict
p = sp.dot(X, Theta.T)
my_predictions = (p[:, 0] + Y_mean).flatten()
idx = sp.argsort(my_predictions).flatten()[::-1][0:10]

print('Top recommendations for you:')
for i in idx:
    name = movie_list[i]
    pr = sp.around(my_predictions[i], 2)
    print("Predicting rating {pr} for movie {name}".format(**locals()))
