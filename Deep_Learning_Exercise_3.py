import numpy as np

# Define the weights and input
w1 = np.array([1.0, 2.0, 3.0])
w2 = np.array([4.0, 5.0])
x = np.array([1.0, 2.0, 3.0])
y = 10

def ReLu(x):
    return np.maximum(0.0, x)

def ReLu_derivative(x):
    return np.where(x > 0.0, 1.0, 0.0)

def two_layer_network(x, w1, w2):
    z1 = np.dot(x, w1)
    a1 = ReLu(z1)
    z2 = np.dot(a1, w2)
    y_hat = np.sum(z2)  # Linear output
    a2 = ReLu(z2)
    return z1, a1, a2, z2, y_hat

def loss_function(y, y_hat):
    return 0.5 * (y - y_hat) ** 2

def loss_derivative(y, y_hat):
    return -(y - y_hat)

def Backpropagation(x, y, w1, w2, learning_rate=0.01):
    # Forward pass
    z1, a1, a2, z2, y_hat = two_layer_network(x, w1, w2)

    # Compute loss and its gradient
    loss = loss_function(y, y_hat)
    dL_dy_hat = loss_derivative(y, y_hat)

    # Backward pass
    dL_dz2 = dL_dy_hat * ReLu_derivative(z2)
    dL_dw2 = a1 * dL_dz2

    dL_dz1 = np.dot(dL_dz2, w2) * ReLu_derivative(z1)
    dL_dw1 = x * dL_dz1

    # Update weights
    w2 -= learning_rate * dL_dw2
    w1 -= learning_rate * dL_dw1

    return w1, w2, loss

def main():
    # Forward pass
    z1, a1, a2, z2, y_hat = two_layer_network(x, w1, w2)
    print("Forward Pass Results:")
    print("z1 (Layer 1 pre-activation):", z1)
    print("a1 (Layer 1 activation):", a1)
    print("a2 (Layer 2 activation):", a2)
    print("z2 (Layer 2 pre-activation):", z2)
    print("y_hat (Predicted Output):", y_hat)

    # Perform backpropagation
    updated_w1, updated_w2, loss = Backpropagation(x, y, w1, w2, learning_rate=0.1)

    # Forward pass with updated weights
    updated_w1, updated_w2,updated_loss,updated_z, y_hat_updated = two_layer_network(x, updated_w1, updated_w2)
    print("\nUpdated Weights and Loss:")
    print("Updated W1:", updated_w1)
    print("Updated W2:", updated_w2)
    print("Loss:", loss)
    print("Updated y_hat (Predicted Output):", y_hat_updated)

main()
