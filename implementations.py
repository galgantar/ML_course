import numpy as np


def compute_gradient(y, tx, w):
    """Computes the gradient for MSE lossat w.

    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        w: shape=(2, ). The vector of model parameters.

    Returns:
        An array of shape (2, ) (same shape as w), containing the gradient of the loss at w.
    """
    N = len(y)
    e = y - tx @ w
    return -1 / N * tx.T @ e


def compute_loss(y, tx, w):
    """Calculate the loss using MSE.

    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        w: shape=(2,). The vector of model parameters.

    Returns:
        the value of the loss (a scalar), corresponding to the input parameters w.
    """
    e = y - tx @ w
    return (e**2).mean() / 2


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """The Gradient Descent (GD) algorithm.

    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        initial_w: shape=(2, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize

    Returns:
        w: a numpy array of shape (2, ) final model parameters
        loss: a scalar containing the final loss value
    """
    w = initial_w
    for n_iter in range(max_iters):
        g = compute_gradient(y, tx, w)
        loss = compute_loss(y, tx, w)
        w = w - gamma * g

        print(
            "GD iter. {bi}/{ti}: loss={l}, w0={w0}, w1={w1}".format(
                bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]
            )
        )

    return w, compute_loss(y, tx, w)


def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient at w from just few examples n and their corresponding y_n labels.

    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        w: shape=(2, ). The vector of model parameters.

    Returns:
        An array of shape (2, ) (same shape as w), containing the stochastic gradient of the loss at w.
    """

    B_n = len(y)
    e = y - tx @ w
    return -1 / B_n * tx.T @ e


def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.

    Example:

     Number of batches = 9

     Batch size = 7                              Remainder = 3
     v     v                                         v v
    |-------|-------|-------|-------|-------|-------|---|
        0       7       14      21      28      35   max batches = 6

    If shuffle is False, the returned batches are the ones started from the indexes:
    0, 7, 14, 21, 28, 35, 0, 7, 14

    If shuffle is True, the returned batches start in:
    7, 28, 14, 35, 14, 0, 21, 28, 7

    To prevent the remainder datapoints from ever being taken into account, each of the shuffled indexes is added a random amount
    8, 28, 16, 38, 14, 0, 22, 28, 9

    This way batches might overlap, but the returned batches are slightly more representative.

    Disclaimer: To keep this function simple, individual datapoints are not shuffled. For a more random result consider using a batch_size of 1.

    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)  # NUmber of data points.
    batch_size = min(data_size, batch_size)  # Limit the possible size of the batch.
    max_batches = int(
        data_size / batch_size
    )  # The maximum amount of non-overlapping batches that can be extracted from the data.
    remainder = (
        data_size - max_batches * batch_size
    )  # Points that would be excluded if no overlap is allowed.

    if shuffle:
        # Generate an array of indexes indicating the start of each batch
        idxs = np.random.randint(max_batches, size=num_batches) * batch_size
        if remainder != 0:
            # Add an random offset to the start of each batch to eventually consider the remainder points
            idxs += np.random.randint(remainder + 1, size=num_batches)
    else:
        # If no shuffle is done, the array of indexes is circular.
        idxs = np.array([i % max_batches for i in range(num_batches)]) * batch_size

    for start in idxs:
        start_index = start  # The first data point of the batch
        end_index = (
            start_index + batch_size
        )  # The first data point of the following batch
        yield y[start_index:end_index], tx[start_index:end_index]


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """The Stochastic Gradient Descent algorithm (SGD) with batch size 1.

    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        initial_w: shape=(2, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of SGD
        gamma: a scalar denoting the stepsize

    Returns:
        w: a numpy array of shape (2, ) final model parameters
        loss: a scalar containing the final loss value
    """

    w = initial_w

    for n_iter in range(max_iters):
        for batch_y, batch_tx in batch_iter(y, tx, batch_size=1, num_batches=1):
            loss = compute_loss(batch_y, batch_tx, w)
            g = compute_stoch_gradient(batch_y, batch_tx, w)
            w = w - gamma * g

        print(
            "SGD iter. {bi}/{ti}: loss={l}, w0={w0}, w1={w1}".format(
                bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]
            )
        )
    return w, compute_loss(y, tx, w)


def least_squares(y, tx):
    """Calculate the least squares solution.
       returns loss, and optimal weights.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        loss: scalar representing the mean squared error.
    """

    w = np.linalg.solve(tx.T @ tx, tx.T @ y)
    return w, compute_loss(y, tx, w)


def ridge_regression(y, tx, lambda_):
    """implement ridge regression.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        lambda_: scalar.

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        loss: scalar representing the mean squared error.
    """
    N = len(y)
    lambda_prime = 2 * N * lambda_
    w = np.linalg.solve(tx.T @ tx + lambda_prime * np.eye(tx.shape[1]), tx.T @ y)
    return w, compute_loss(y, tx, w)


### --------------------------Logistic Regression--------------------------


def sigmoid(t):
    """Apply sigmoid function on t.

    Args:
        t: numpy array with shape (N, )

    Returns:
        numpy array with shape (N, ) with applied sigmoid function
    """
    return np.where(t >= 0, 
                    1 / (1 + np.exp(-t)), 
                    np.exp(t) / (1 + np.exp(t)))


def calculate_logistic_loss(y, tx, w):
    """Compute the cost by negative log likelihood.

    Args:
        y: shape=(N, )
        tx: shape=(N, D)
        w: shape=(D, ). The vector of model parameters.

    Returns:
        scalar: the value of the loss, corresponding to the input parameters w.
    """

    N = len(y)
    z = tx @ w
    loss = -1/N * np.sum(y * z - np.logaddexp(0, z))
    return loss


def calculate_logistic_gradient(y, tx, w):
    """Compute the gradient of loss.

    Args:
        y: shape=(N, )
        tx: shape=(N, D)
        w: shape=(D, ). The vector of model parameters.

    Returns:
        An array of shape (D, ) (same shape as w), containing the gradient of the loss at w.
    """

    N = len(y)
    z = tx @ w
    pred = sigmoid(z)
    gradient = 1/N * tx.T @ (pred - y)
    return gradient


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Logistic regression.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        initial_w: numpy array of shape (D,), D is the number of features.
        max_iters: a scalar denoting the total number of iterations of logistic regression.
        gamma: a scalar denoting the stepsize.

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        loss: scalar, the final loss value.
    """
    w = initial_w
    
    for n_iter in range(max_iters):
        gradient = calculate_logistic_gradient(y, tx, w)
        
        w = w - gamma * gradient
        loss = calculate_logistic_loss(y, tx, w)
        
        if n_iter % 100 == 0 or n_iter == max_iters - 1:
            print(f"Logistic regression iter. {n_iter}/{max_iters - 1}: loss={loss}")
    
    return w, calculate_logistic_loss(y, tx, w)


def calculate_reg_logistic_loss(y, tx, w, lambda_):
    """Compute the cost by negative log likelihood with L2 regularization.

    Args:
        y: shape=(N, )
        tx: shape=(N, D)
        w: shape=(D, ). The vector of model parameters.
        lambda_: scalar, the regularization parameter.

    Returns:
        scalar: the value of the loss (a scalar), corresponding to the input parameters w.
    """

    N = len(y)
    z = tx @ w
    nll = -1/N * np.sum(y * z - np.logaddexp(0, z))    
    reg_term = lambda_ * np.sum(w**2)
    return nll + reg_term


def calculate_reg_logistic_gradient(y, tx, w, lambda_):
    """Compute the gradient of loss with L2 regularization.

    Args:
        y: shape=(N, )
        tx: shape=(N, D)
        w: shape=(D, ). The vector of model parameters.
        lambda_: scalar, the regularization parameter.

    Returns:
        An array of shape (D, ) (same shape as w), containing the gradient of the loss at w.
    """

    N = len(y)
    z = tx @ w
    pred = sigmoid(z)
    
    nll_gradient = 1/N * tx.T @ (pred - y)    
    reg_gradient = 2 * lambda_ * w
    
    return nll_gradient + reg_gradient


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Regularized logistic regression.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        lambda_: scalar, the regularization parameter.
        initial_w: numpy array of shape (D,), D is the number of features.
        max_iters: a scalar denoting the total number of iterations of logistic regression.
        gamma: a scalar denoting the stepsize.

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        loss: scalar, the final loss value (unregularized).
    """

    w = initial_w
    
    for n_iter in range(max_iters):
        gradient = calculate_reg_logistic_gradient(y, tx, w, lambda_)
        
        w = w - gamma * gradient
        
        reg_loss = calculate_reg_logistic_loss(y, tx, w, lambda_)
        
        if n_iter % 100 == 0 or n_iter == max_iters - 1:
            print(f"Regularized logistic regression iter. {n_iter}/{max_iters - 1}: loss={reg_loss}")
    
    return w, calculate_logistic_loss(y, tx, w)