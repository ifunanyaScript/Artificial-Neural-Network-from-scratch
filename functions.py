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

# ifunanyaScript
