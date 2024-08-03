# #Exercises

# 1. Consider the following two networks. W is a matrix, x is a vector, z is a vector, and a is a
# vector. y^ is a scalar and a final prediction. Initialize x, w randomly, z is a dot product of x
# and w, a is ReLU(z). Initialize X and W randomly

#2. Implement forward pass for the above two networks. Print activation values for each
# neuron at each layer. Print the loss value (y^).

# #3. Implement the forward pass using vectorized operations, i.e. W should be a matrix, x, z
# and a are vectors. The implementation should not contain any loops.


import numpy as np

#W matrix
w1 = np.random.rand(3)
w2 = np.random.rand(3)

#X
x = np.random.rand(3)

def ReLu(x):
    ReLu = np.maximum(0,x)
    return ReLu

def two_layer_network(x,w1,w2):
    # Z is a dot product of a and w
    z1 = np.dot(x,w1)
    a1 = ReLu(z1)
    z2 = np.dot(a1,w2)
    y_hat = z2
    a2 = ReLu(z2)
    return z1,a1,a2,z2,y_hat

def main():
    ReLu(x)
    two_layer_network(x, w1, w2)
    z2_1,a2_1,a2_2,z2_2,y_hat = two_layer_network(x,w1,w2)
    print("2 Layered Network :-")
    print("Values of X1,X2,X3 --- Layer 1 :",x)
    print("Value of W1 --- :",w1)
    print("Value of W2 --- :",w2)
    print("Value of A(The ReLu Function for 1st Layer) :",a2_1)
    print("Value of A(The ReLu Function for 2nd Layer) :", a2_2)
    print("Final value :",y_hat)


main()