import numpy as np
from scipy.optimize import minimize, LinearConstraint, BFGS
from matplotlib import pyplot as plt
from sklearn.datasets.samples_generator import make_blobs, make_moons

# fast (but doesn't work for any type of kernel)
# write here the function you are going to optimize
# note that we are using the optimizer which minimizes the function, so you need to account for this
# returns value of the function calculated based on parameters
"""unfold parameters description below"""
# alphas - 1D array
# y - labels
# x - data
# kernel - callable, defines kernel function to use
def function_to_optimize(alphas, y, x, kernel):
    alphas = alphas.reshape(len(alphas), 1)
    y = y.reshape(len(y), 1)
    y_n = np.dot(y, y.T)
    alphas_n = np.dot(alphas, alphas.T)
    x_n = kernel(x, x)
    func = np.sum(alphas) - 0.5 * np.sum(y_n * alphas_n * x_n)
    return -func


# slow but generalisable
def function_to_optimize_1(alphas, y, x, kernel):
    n = len(alphas)
    term_2 = 0
    for i in range(n):
        for j in range(n):
            term_2 += alphas[i] * alphas[j] * y[i] * y[j] * kernel(x[i, :], x[j, :])
    func = np.sum(alphas) - 0.5 * term_2
    return -func


# linear kernel <x,z>
# returns scalar - output of the kernel
# x, z - vectors / matrices
def linear_kernel(x, z):
    return np.dot(x, z.T)


# polynomial kernel (<x,z> + 1)^3 with degree = 3 and c = 1
# returns scalar - output of the kernel
# x, z - vectors / ndarrays
def polynomial_kernel(x, z):
    dot_product = np.dot(x, z.T)
    dot_product = (dot_product + 1) ** 3
    return dot_product


# returns array of indices of all support vectors, i.e. those for which alpha > 0
# instead of zero we are using some threshold to account for roundoff errors
# alphas - array of solutions
# thresh - threshold
def find_support_vector_inds(alphas, thresh):
    return np.where(alphas > thresh)[0]


# calculates w and b after alphas are found. w is calculated only if we are using linear kernel
# w is a vector, b is a scalar
# for calculating b use this formula: b = - (wx_pos + wx_neg)/2
# x_pos and x_neg are positive and negative support vectors lying *exactly* on the margin
# Recall that if sample is on correct side, then alpha = 0, if it's exactly on the margin, then 0 < alpha < C,
# if it's on wrong side, alpha = C. So, chose any of those for which 0 < alpha < C, accounting for roundoff errors (!)
"""unfold parameters description below"""
# alphas - array of solutions
# y - labels
# x - data
# sv_inds - indices of support vectors
# kernel - callable, defines kernel function to use
# thresh - threshold
# C - constant
def find_w_b(alphas, y, x, sv_inds, kernel, thresh, C):
    w = None
    # find w if it's a linear case
    if kernel == linear_kernel:
        w = np.zeros(x[0, :].shape)
        for i in sv_inds:
            w += alphas[i] * y[i] * x[i, :]
    # find b. first identify indexes of support vectors lying on the margin (filter from errors)
    # we need only one positive and one negative support vector
    pos_sv_ind = ((y == 1) & (alphas > thresh) & (alphas < (C - 1))).nonzero()[0][0]
    neg_sv_ind = ((y == -1) & (alphas > thresh) & (alphas < (C - 1))).nonzero()[0][0]
    # calculate w*x_pos
    w_pos = 0
    for i in sv_inds:
        w_pos += alphas[i] * y[i] * (kernel(x[i, :], x[pos_sv_ind, :]))
    # calculate w*x_neg
    w_neg = 0
    for i in sv_inds:
        w_neg += alphas[i] * y[i] * (kernel(x[i, :], x[neg_sv_ind, :]))
    # take -average. to understand why refer here - https://stats.stackexchange.com/questions/211310/deriving-the-intercept-term-in-a-linearly-separable-and-soft-margin-svm
    b = -(w_pos + w_neg) / 2
    return w, b


# make predictions for x_test
# returns array of predictions (1 or -1)
"""unfold parameters description below"""
# alphas - array of solutions
# y - labels
# x - data
# x_test - data for which predictions should be made
# b - calculated b
# sv_inds - indices of support vectors
# kernel - callable, defines kernel function to use
def predict(alphas, y, x, x_test, b, sv_inds, kernel):
    predictions = np.zeros(shape=(len(x_test),))
    for j, sample in enumerate(x_test):
        pred = b
        for i in sv_inds:
            pred += alphas[i] * y[i] * (kernel(x[i, :], sample))
        predictions[j] = 1 if pred > 0 else -1
    return predictions


# this is the main function which brings it all together
# examine it to understand what's happening in each line
# you only need to finish two lines of code here, describing the constraints
# don't change anything else (!)
def minimize_and_plot(X, Y, kernel, C, thresh):
    n = len(Y)
    # arguments to pass to minimize function
    args = (Y, X, kernel)

    # define the constraints (page 20) as instances of scipy.optimize.LinearConstraint
    # constraints each alpha to be from 0 to C
    alpha_constr = LinearConstraint(np.identity(n), lb=0, ub=C)
    # constraints sum of (alpha * y)
    alpha_y_constr = LinearConstraint(Y, lb=0, ub=0)

    print("Starting computations...")
    # minimization. we are using ready QP solver 'trust-constr'
    result = minimize(fun=function_to_optimize, method='trust-constr', x0=np.empty(shape=(n,)), jac='2-point',
                      hess=BFGS(exception_strategy='skip_update'), constraints=[alpha_constr, alpha_y_constr],
                      args=args)
    # prints the results. If status==0, then the optimizer failed to find the optimal value
    print("status:", result.status)
    print("message:", result.message)

    alphas = result.x
    # indexes of support vectors
    sv_inds = find_support_vector_inds(alphas, thresh)
    print("alphas of support vectors:", '\n', alphas[sv_inds])

    w, b = find_w_b(alphas, Y, X, sv_inds, kernel, thresh, C)

    # create a mesh to plot points and predictions
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xrange, yrange = np.meshgrid(np.arange(x_min, x_max, 0.02),
                                 np.arange(y_min, y_max, 0.02))
    # form a grid by taking each point point from x and y range
    grid = np.c_[xrange.ravel(), yrange.ravel()]
    grid = grid.astype(float)
    # make predictions for each point of the grid
    grid_predictions = predict(alphas, Y, X, grid, b, sv_inds, kernel)
    grid_predictions = grid_predictions.reshape(xrange.shape)

    # plot color grid points according to the prediction made for each point
    plt.contourf(xrange, yrange, grid_predictions, cmap='copper', alpha=0.8)
    # plot initial data points
    plt.scatter(X[:, 0], X[:, 1], c=Y, s=50, cmap='autumn')
    # plot support vectors
    plt.scatter(X[sv_inds, 0], X[sv_inds, 1], s=3, c="black")

    if w is not None:  # print lines on which support vectors should reside
        x_plot = np.linspace(x_min, x_max - 0.02, 1000)
        y_plot_1 = (- w[0] * x_plot - b + 1) / w[1]
        y_plot_2 = (- w[0] * x_plot - b - 1) / w[1]
        plt.plot(x_plot, y_plot_1)
        plt.plot(x_plot, y_plot_2)

    plt.title('SVM Results ' + kernel.__name__)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


# here we assign values for constants
# you can change them when you are testing your solution but when submitting leave it in original state
n_samples = 50
C_const = 100
threshold = 1e-3

# generating (almost) linearly separable data, replacing 0 labels with -1
X_blob, Y_blob = make_blobs(n_samples=n_samples, centers=2, random_state=0, cluster_std=1.00)
Y_blob[Y_blob == 0] = -1
# plt.scatter(X_blob[:, 0], X_blob[:, 1], c=Y_blob, s=50, cmap='autumn')
# plt.show()
minimize_and_plot(X_blob, Y_blob, linear_kernel, C_const, threshold)  # svm with linear kernel
minimize_and_plot(X_blob, Y_blob, polynomial_kernel, C_const, threshold)  # svm with polynomial kernel

# generating moon-shaped data, replacing 0 labels with -1
X_moon, Y_moon = make_moons(n_samples=n_samples, shuffle=False, noise=0.10, random_state=0)
Y_moon[Y_moon == 0] = -1
# plt.scatter(X_moon[:, 0], X_moon[:, 1], c=Y_moon, s=50, cmap='autumn')
# plt.show()
minimize_and_plot(X_moon, Y_moon, linear_kernel, C_const, threshold)  # svm with linear kernel
minimize_and_plot(X_moon, Y_moon, polynomial_kernel, C_const, threshold)  # svm with polynomial kernel
