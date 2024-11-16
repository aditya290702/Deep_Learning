# Create an Autoencoder class with Linear-ReLU-Linear-ReLU for both encoder and decoder.
# Use this class to generate a latent representation for an input, X.
# X is a 784 dimensional vector. Hidden dimension size is 128 and the latent dimension size is 32.
# You just need to implement a forward pass and no need to train the network.
# Generate a single input sample of size 784-dim vector at random and generate the latent representation for this input vector.

import torch
from torch import nn
import numpy as np


def encoder_decoder():
    input_size = 784
    hidden_size = 128
    latent_size = 32

    #my architecture for encoder
    encoder = nn.Sequential(nn.Linear(input_size,hidden_size),
                            nn.ReLU(),
                            nn.Linear(hidden_size,latent_size),
                            nn.ReLU())

    #my architecture for decoder
    decoder = nn.Sequential(nn.Linear(latent_size,hidden_size),
                            nn.ReLU(),
                            nn.Linear(hidden_size,input_size),
                            nn.ReLU())

    #i am generating random samples of the input size that will later be passed on to the encoder
    x = torch.randn(1,input_size)

    #passing x through the encoder for a latent rep.
    latent_representation = encoder(x)

    #passsing the latent representation to the decoder architecture trying to reconstruct the inputs
    reconstructed = decoder(latent_representation)

    #printhing the encoded values
    print("----------------------------------------------------------")
    print("latent_representation")
    print("----------------------------------------------------------")
    print(latent_representation)
    print()

    #printing the reconstructed values
    print("----------------------------------------------------------")
    print("reconstructed from encoder representation")
    print("----------------------------------------------------------")
    print(reconstructed)
    print("----------------------------------------------------------")
    print("shape of reconstructed representation")
    print("----------------------------------------------------------")
    print(np.shape(reconstructed))
    # #trying to check how accurate my valyes have been generate wrt the input
    # loss = (np.std(x.numpy()) - np.std(reconstructed.detach()   .numpy()))
    # print(f"the loss in reconstruction {loss}")


def main():
    encoder_decoder()

if __name__ == '__main__':
    main()