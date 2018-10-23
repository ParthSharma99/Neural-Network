import numpy as np


"""
Functions:
-> initialize_parameters
-> linear_forward
-> linear_forward_activation
-> linear_backward
-> linear_backward_activation

"""


def sigmoid(Z):
    val = 1/(1 + np.exp(-Z))
    return val


def relu(Z):
    return np.maximum(Z, 0)


def relu_backward(dA, activation_cache):
    return dA*(activation_cache > 0)


def sigmoid_backward(dA, activation_cache):
    return dA*(1 - activation_cache**2)


def initialize_parameters(layer_dims):
    """
    :param layer_dims: list containing sizes of input layer
    :return: parameters ( a python dictionary) containing W1,b1,W2,b2
    """
    parameters = {}
    L = len(layer_dims)

    for l in range(1, L):
        # Initialising
        parameters["W" + str(l)] = np.random.randn(layer_dims[l],layer_dims[l-1]) * 0.01
        parameters["b" + str(l)] = np.zeros((layer_dims[l],1))
        # Asserting shapes
        assert (parameters["W" + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
        assert (parameters["b" + str(l)].shape == (layer_dims[l], 1))
    return parameters


def linear_forward(A, W, b):
    """
    :param A: activation from previos layer
    :param W: weights matrix
    :param b: biases matrix
    :return:
    Z : input of activation function
    cache : tuple containing previous parameters
    """
    try:
        Z = np.dot(W, A) + b
    except ValueError:
        Z = np.dot(W, A.T) + b


    cache = (A, W, b)

    return Z, cache


def linear_forward_activation(A_prev, W, b, activation):
    """
    :param A_prev: activation matrix from previous layer
    :param W: weight matrix
    :param b: biases matrix
    :param activation:
    :return:
    """
    if activation == "sigmoid":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A = sigmoid(Z)
        activation_cache = Z
    elif activation == "relu":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A = relu(Z)
        activation_cache = Z

    cache = (linear_cache, activation_cache)
    return A, cache


def forward_model(X, parameters):

    caches = []
    A = X
    L = len(parameters)//2
    for l in range(1, L):
        A_prev = A
        A, cache = linear_forward_activation(A_prev, parameters["W" + str(l)], parameters["b" + str(l)], "relu")
        caches.append(cache)

    AL, cache = linear_forward_activation(A, parameters["W" + str(L)], parameters["b" + str(L)], "sigmoid")
    caches.append(cache)
    assert (AL.shape == (1, X.shape[1]))

    return AL, caches


#cost = (−1/m)(y.log(a[L])+(1−y).log(1−a[L]))
def compute_cost(Y, AL):

    m = Y.shape[1]
    cost = -np.sum((np.dot(Y, np.log(AL)) + np.dot(1-Y, np.log(1-AL))))/m
    cost = np.squeeze(cost)
    assert (cost.shape == ())
    return cost


def linear_backward(dZ, cache):
    """
    :param cache: contains parameters
    :return:
    gradients of parameters
    """
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = np.dot(dZ, A_prev.T)/m
    db = np.sum(dZ, axis=1, keepdims=True)/m
    dA_prev = np.dot(W, dZ)

    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)

    return dA_prev, dW, db


def linear_backward_activation(dA, cache, activation):
    """
    :param A_prev: activation matrix from previous layer
    :param W: weight matrix
    :param b: biases matrix
    :param activation:
    :return:
    """
    linear_cache, activation_cache = cache
    if activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, cache)
    elif activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, cache)

    return dA_prev, dW, db


def backward_model(caches, AL, Y):
    """
    :param caches: list containing cache from linear_forward() sigmoid for [L-1] else relu for 1,2,3...L-2
    :param AL:probability vector; output of forward propagation
    :param Y: true "label" vector
    :return:
    grads (a dictionary with gradients)
    """
    grads = {}
    L = len(caches)
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)

    dAL = -(np.divide(Y,AL) - np.divide(1-Y,1-AL))

    current_cache = linear_backward_activation(dAL, caches[L-1], "sigmoid")
    grads["dA" + str(L)], grads["dW" + str(L), grads["db" + str(L)]] = current_cache

    for l in range(L-2, 0, -1):
        current_cache = linear_backward_activation(grads["dA" + str(l+2)], caches[l], "relu")
        grads["dA" + str(l+1)], grads["dW" + str(l+1), grads["db" + str(l+1)]] = current_cache

    return grads


def update_parameters(parameters, grads, learning_rate):
    L = len(parameters)//2

    for l in range(1,L+1):
        parameters["W" + str(l)] += -learning_rate*grads["dW" + str(l)]
        parameters["b" + str(l)] += -learning_rate * grads["db" + str(l)]

    return parameters


def nn_model(X, Y, layer_dims, learning_rate= 0.01, num_iterations = 3000):

    parameters = initialize_parameters(layer_dims)
    costs = []

    for i in range(num_iterations):

        AL, caches = forward_model(X, parameters)

        cost = compute_cost(Y,AL)
        costs.append(cost)
        grads = backward_model(caches, AL, Y)
        parameters = update_parameters(parameters,grads,learning_rate)

    return parameters
