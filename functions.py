# Randomly generates the initial weights and biases.
def initial_parameters():
    first_weights = np.random.rand(10, 784) - .5
    first_bias = np.random.rand(10, 1) - .5
    second_weights = np.random.rand(10, 10) - .5
    second_bias = np.random.rand(10, 1) - .5
    return first_weights, first_bias, second_weights, second_bias

# # Leaky ReLU activation.
# def Leaky_ReLU(step_1, alpha):
#     for sub_array in step_1:
#         for i, value in enumerate(sub_array):
#             if value>0:
#                 sub_array[i] = value
#             else:
#                 sub_array[i] = (alpha*value)
#     return step_1

# # Derivative of Leaky ReLU activation.
# def LR_deriv(step_1, alpha):
#     for sub_array in step_1:
#         for i, value in enumerate(sub_array):
#             if value>0:
#                 sub_array[i] = 1
#             else:
#                 sub_array[i] = alpha
#     return step_1


# ReLU activation.
def ReLU(Z):
    return np.maximum(Z, 0)

# Derivative of ReLU actiavtion.
def ReLU_deriv(Z):
    return Z > 0


# Softmax activation for output layer.
def Softmax(step_3):
    prob_array = np.exp(step_3) / sum(np.exp(step_3))
    return prob_array


# One-Hot encoder for target labels.
def One_Hot_Encoder(Y_train):
    binary_labels = np.zeros((Y_train.size, Y_train.max() + 1))
    binary_labels[np.arange(Y_train.size), Y_train] = 1
    binary_labels = binary_labels.T
    return binary_labels
