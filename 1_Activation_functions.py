# #Implement the following functions in Python from scratch. Do not use any library
# functions. You are allowed to use numpy and matplotlib. Generate 100 equally spaced
# values between -10 and 10. Call this list as z. Implement the following functions and its
# derivative. Use class notes to find the expression for these functions. Use z as input and
# plot both the function outputs and its derivative outputs. Upload your code into Github
# and share it with me.
# a. Sigmoid
# b. Tan h
# c. ReLU (Rectified Linear Unit)
# d. Leaky ReLU
# e. Softmax

# 2. Write down the observations from the plot for all the above functions in the code.
# a. What are the min and max values for the functions
# b. Are the output of the function zero-centred
# c. What happens to the gradient when the input values are too small or too big
#-----------------------------------------------------------------------------------------------------------------------

import numpy as np
from matplotlib import pyplot as plt

# Generate 100 equally spaced values between -10 and 10.
z = np.linspace(-10, 10, 100)
print("Z values")
print(z)
print()
print()
print("-------------------------------------------------------------------------------")

fig, ax = plt.subplots(2,2, figsize = (5,10))

def sigmoid_derivatives(z):

    # now implementing sigmoid function
    sigmoid = 1 / (1 + np.exp(-z))
    sigmoid_derivative = sigmoid * (1 - sigmoid)

    # Plotting the sigmoid function
    plt.plot(z, sigmoid,label = 'sigmoid')
    plt.plot(z, sigmoid_derivative, label='sigmoid_derivative')
    plt.xlabel("x")
    plt.ylabel("np values")
    plt.title("Sigmoid")
    plt.tight_layout()
    plt.legend()
    print()
    print()
    print("SIGMOID FUNC. VALUES : ",sigmoid)
    print()
    print()
    print("a. What are the min and max values for the functions")
    print()
    print("Sigmoid min :", np.min(sigmoid))
    print("Sigmoid max :", np.max(sigmoid))
    print()
    print()
    print("b. Are the output of the function zero-centred")
    print()
    Sig_mean = np.mean(sigmoid)
    if(Sig_mean<0.1):
        print(Sig_mean)
        print("It is a 0-Centered Function")
        print()
        print()
    else:
        print(Sig_mean)
        print("Not a 0-Centered Function")
        print()
        print()
    print("----------------------------------------------------------------------------------------")


def Tanh_Derivatives(z):

    # Implementing the Tanh funciton
    Tanh = ((np.exp(z)) - (np.exp(-z)))/((np.exp(-z)) + (np.exp(z)))
    Tanh_derivative = 1 - (Tanh)**2
    print()
    print()
    print("Tanh FUNC. VALUES : ", Tanh)
    print()
    print()
    print("a. What are the min and max values for the functions")
    print()
    print("Tanh min :", np.min(Tanh))
    print("tanh max :", np.max(Tanh))
    print()
    print()
    print("b. Are the output of the function zero-centred")
    print()
    Tanh_mean = np.mean(Tanh)
    if (Tanh_mean < 1):
        print(Tanh_mean)
        print("It is a 0-Centered Function")
        print()
        print()
    else:
        print(Tanh_mean)
        print("Not a 0-Centered Function")
        print()
        print()
    print("-------------------------------------------------------------------------------")


    # Plotting the Tan h function
    plt.plot(z, Tanh,label = 'Tanh')
    plt.plot(z, Tanh_derivative,label = 'Tan h_Derivative')
    plt.xlabel("x")
    plt.ylabel("np values")
    plt.title("Tanh")
    plt.tight_layout()
    plt.legend()

def ReLu_Derivatives(z):

    # Implementing the ReLu function
    ReLu = np.maximum(0,z)
    ReLu_derivative = np.where(z > 0, 1, 0)


    # Plotting the function
    plt.plot(z, ReLu, label='RELU')
    plt.plot(z, ReLu_derivative, label='RELU_Derivative')
    plt.xlabel("x")
    plt.ylabel("np values")
    plt.title("ReLu FUNC. VALUES")
    plt.tight_layout()
    plt.legend()
    print()
    print()
    print("ReLu : ", ReLu)
    print()
    print()
    print("a. What are the min and max values for the functions")
    print()
    print("ReLu min :", np.min(ReLu))
    print("ReLu max :", np.max(ReLu))
    print()
    print()
    print("b. Are the output of the function zero-centred")
    print()
    ReLu_mean = np.mean(ReLu)
    if (ReLu_mean < 1):
        print(ReLu_mean)
        print("It is a 0-Centered Function")
        print()
        print()
    else:
        print(ReLu_mean)
        print("Not a 0-Centered Function")
        print()
        print()
    print("-------------------------------------------------------------------------------")



def Softmax(z):

    # Implementing the sOFTMAX function
    print("Softmax : ")


def main():
    plt.subplot(2, 2, 1)
    sigmoid_derivatives(z)
    plt.grid(True)
    plt.tight_layout()

    plt.subplot(2, 2, 2)
    Tanh_Derivatives(z)
    plt.grid(True)
    plt.tight_layout()

    plt.subplot(2, 2, 3)
    ReLu_Derivatives(z)
    plt.grid(True)
    plt.tight_layout()

    plt.grid(True)
    plt.tight_layout()
    plt.show()

main()
