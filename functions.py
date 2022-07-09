import numpy as np

def initial_parameters():
    first_weights = np.random.rand(10, 784) - .5
    first_bias = np.random.rand(10, 1) - .5
    second_weights = np.random.rand(10, 10) - .5
    second_bias = np.random.rand(10, 1) - .5
    return first_weights, first_bias, second_weights, second_bias

def Leaky_ReLU(step_1, alpha):
    for sub_array in step_1:
        for i, value in enumerate(sub_array):
            if value>0:
                sub_array[i] = value
            else:
                sub_array[i] = (alpha*value)
    return step_1

def LR_deriv(step_1, alpha):
    for sub_array in step_1:
        for i, value in enumerate(sub_array):
            if value>0:
                sub_array[i] = 1
            else:
                sub_array[i] = alpha
    return step_1

def Softmax(step_3):
    prob_array = np.exp(step_3) / sum(np.exp(step_3))
    return prob_array

def One_Hot_Encoder(Y_train):
    binary_labels = np.zeros((Y_train.size, Y_train.max() + 1))
    binary_labels[np.arange(Y_train.size), Y_train] = 1
    binary_labels = binary_labels.T
    return binary_labels

def forward_pass(first_weights, first_bias, second_weights, second_bias, X_train):
    step_1 = first_weights.dot(X_train) + first_bias
    step_2 = Leaky_ReLU(step_1, alpha=0.01)
    step_3 = second_weights.dot(step_2) + second_bias
    step_4 = Softmax(step_3)
    return step_1, step_2, step_3, step_4

# ifunanyaScript
