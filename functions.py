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


# Forward propagation.
def forward_pass(first_weights, first_bias, second_weights, second_bias, X_train):
    step_1 = first_weights.dot(X_train) + first_bias
    step_2 = ReLU(step_1)
    step_3 = second_weights.dot(step_2) + second_bias
    step_4 = Softmax(step_3)
    return step_1, step_2, step_3, step_4


# Backward propagation.
def backward_pass(step_1, step_2, step_3, step_4, first_weights, second_weights, X_train, Y_train):
    binary_labels = One_Hot_Encoder(Y_train)
    d_step_3 = (step_4 - binary_labels)
    d_second_weights = 1/samples * d_step_3.dot(step_2.T)
    d_second_bias = 1/samples * np.sum(d_step_3)
    d_step_2 = second_weights.T.dot(d_step_3) * ReLU_deriv(step_1)
    d_first_weights = 1/samples * d_step_2.dot(X_train.T)
    d_first_bias = 1/samples * np.sum(d_step_2)
    return d_first_weights, d_first_bias, d_second_weights, d_second_bias


# Update function for weights and biases.
def update_parameters(first_weights, first_bias, second_weights, second_bias,
                      d_first_weights, d_first_bias, d_second_weights, d_second_bias, 
                      learning_rate):
    first_weights = first_weights - d_first_weights*learning_rate
    first_bias = first_bias - d_first_bias*learning_rate
    second_weights = second_weights - d_second_weights*learning_rate
    second_bias = second_bias - d_second_bias*learning_rate
    return first_weights, first_bias, second_weights, second_bias


# # Get neural network's predictions.
# def predictions(step_4):
#     return np.argmax(step_4, 0)

# Measure network's accuracy.
def accuracy(predictions, Y_train):
    return np.sum(predictions == Y_train)/Y_train.shape[0]


# Gradient for training the network.
def gradient_descent(X_train, Y_train, learning_rate, epochs):
    first_weights, first_bias, second_weights, second_bias = initial_parameters()
    for epoch in range(epochs):
        step_1, step_2, step_3, step_4 = forward_pass(first_weights, first_bias,
                                                      second_weights, second_bias, X_train)
        
        d_first_weights, d_first_bias, d_second_weights, d_second_bias = backward_pass(
                                                                        step_1, step_2, 
                                                                        step_3, step_4,
                                                                        first_weights,
                                                                        second_weights, 
                                                                        X_train, Y_train)
        
        first_weights, first_bias, second_weights, second_bias = update_parameters(
                                                            first_weights, first_bias, 
                                                            second_weights, second_bias,
                                                            d_first_weights, d_first_bias, 
                                                            d_second_weights, d_second_bias, 
                                                            learning_rate)

        if epoch % 100 == 0:
            print(f"Epoch: {epoch}/{epochs}")
            pred = predictions(step_4)
            print(f"Accuracy: {(accuracy(pred, Y_train)*100):.2f}%\n_________________")
    return first_weights, first_bias, second_weights, second_bias


# Modified `predictions` to contain confidence.
def predictions(step_4):
    pred = np.argmax(step_4, 0)
    confidence = round(100 * (np.max(step_4)), 2)
    return pred, confidence

# NN prediction.
def test(X, first_weights, first_bias, second_weights, second_bias):
    _, _, _, step_4 = forward_pass(first_weights, first_bias, second_weights, second_bias, X)
    pred, confidence = predictions(step_4)
    return pred, confidence

# Validating the prediction.
def test_validation(index, first_weights, first_bias, second_weights, second_bias):
    image_sample = X_train[:, index, None]
    pred, confidence = test(image_sample, first_weights, first_bias, second_weights, second_bias)
    label = Y_train[index]
    
    image_sample = image_sample.reshape((28, 28)) * 255
    plt.imshow(image_sample, interpolation="nearest", cmap="gray")
    plt.title(f"Model's prediction: {pred}\nConfidence: {confidence}%\nActual label: {label}")
    plt.axis("off")
    plt.show()

 
# This function predicts the number using the trained or optimised parameters(weights and biases).
def predict_number(index, first_weights, first_bias, second_weights, second_bias):
    image_sample = unlabelled_data[:, index, None]/255
    pred, confidence = test(image_sample, first_weights, first_bias, second_weights, second_bias)
    
    image_sample = image_sample.reshape((28, 28)) * 255
    plt.imshow(image_sample, interpolation="nearest", cmap="gray")
    plt.title(f"Model's prediction: {pred}\nConfidence: {confidence}%")
    plt.axis("off")
    plt.show()
    
# ifunanyaScript
