import numpy as np 

input_neurons = 2 
hidden_neurons = 4 
output_neurons = 2 
iterations = 6000

# Initialization
input_data = np.random.randint(1, 5, input_neurons)
output_data = np.array([1.0, 0.0])
hidden_weights = np.random.rand(input_neurons, hidden_neurons)
output_weights = np.random.rand(hidden_neurons, output_neurons)
hidden_bias = np.random.rand(1, hidden_neurons)
output_bias = np.random.rand(1, output_neurons)

# Activation function (sigmoid) and its gradient
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Training
for i in range(iterations):
    hidden_layer = sigmoid(np.dot(input_data, hidden_weights) + hidden_bias)
    output_layer = sigmoid(np.dot(hidden_layer, output_weights) + output_bias)

    error = output_data - output_layer 
    gradient_output_layer = output_layer * (1 - output_layer)  # Simplified sigmoid gradient
    error_terms_output = gradient_output_layer * error 
    error_terms_hidden = hidden_layer * (1 - hidden_layer) * np.dot(error_terms_output, output_weights.T)

    hidden_weights += 0.05 * np.outer(input_data, error_terms_hidden)
    output_weights += 0.05 * np.outer(hidden_layer, error_terms_output)

    if i < 50 or i > iterations - 50:
        print("********") 
        print("iteration:", i, "::::", error) 
        print("###output########", output_layer)
