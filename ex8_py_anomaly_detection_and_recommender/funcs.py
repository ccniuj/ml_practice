import scipy as sp
from numpy.linalg import det, inv
from functools import reduce
import codecs

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

def cofi_cost_function(Y, R, num_users, num_movies, num_features, _lambda):
    def compute(params):
        X = params[0:(num_movies*num_features)]. \
              reshape([num_movies, num_features])
        Theta = params[(num_movies*num_features):]. \
                  reshape([num_users, num_features])
        err = sp.dot(X, Theta.T) * R - Y
        J = (1 / 2) * sp.sum(err ** 2) + (_lambda / 2) * (sp.sum(Theta ** 2) + sp.sum(X ** 2))
        return J
    return compute

def cofi_gradients(Y, R, num_users, num_movies, num_features, _lambda):
    def compute(params):
        X = params[0:(num_movies*num_features)]. \
              reshape([num_movies, num_features])
        Theta = params[(num_movies*num_features):]. \
                  reshape([num_users, num_features])
        err = sp.dot(X, Theta.T) * R - Y
        X_grad = sp.dot(err, Theta) + _lambda * X
        Theta_grad = sp.dot(X.T, err).T + _lambda * Theta
        grad = sp.hstack((X_grad.flatten(), Theta_grad.flatten()))
        return grad
    return compute

def check_gradients(_lambda=0):
    X_t = sp.random.rand(4, 3)
    Theta_t = sp.random.rand(5, 3)
    Y = sp.dot(X_t, Theta_t.T)
    Y[sp.random.rand(Y.shape[0], Y.shape[1]) > 0.5] = 0
    R = sp.zeros(Y.shape)
    R[Y != 0] = 1

    X = sp.random.rand(X_t.shape[0], X_t.shape[1])
    Theta = sp.random.rand(Theta_t.shape[0], Theta_t.shape[1])
    num_users = Y.shape[1]
    num_movies = Y.shape[0]
    num_features = Theta_t.shape[1]

    params = sp.hstack((X.flatten(), Theta.flatten()))

    D = cofi_gradients(Y, R, num_users, num_movies, num_features, _lambda)(params)
    N = compute_numerical_gradient(cofi_cost_function(Y, R, num_users, num_movies, num_features, _lambda), params)
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

def load_movie_list():
    data = codecs.open('movie_ids.txt', encoding="cp1252").read()
    return sp.array([' '.join(e.split(' ')[1:]) for e in data.split('\n')][0:-1])

def normalize_ratings(Y, R):
    Y_mean = sp.array([sp.sum(Y, 1) / sp.sum(R, 1)])
    Y_norm = Y - Y_mean.T * R
    return [Y_norm, Y_mean]
