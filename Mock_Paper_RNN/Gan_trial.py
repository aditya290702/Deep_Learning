# The generator should have Linear - ReLU - Linear - ReLU - Linear layers.
# The discriminator should have Linear - LeakyRelu - Linear - LeakyRelu - Linear - Sigmoid.
# input_dim = 10  # Dimension of the input vector (input to the generator)
# hidden_dim = 128  # Hidden layer size
# output_dim = 1  # Output dimension (1D data point for simplicity)
# batch_size = 64
# num_epochs = 5000
# learning_rate = 0.0002
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
import numpy as np

input_dim = 10
hidden_dim = 120
output_dim = 1
batch_size = 64
num_epochs = 5000
learning_rate = 0.0002

#generator arch
generator = nn.Sequential(nn.Linear(input_dim,hidden_dim),
                          nn.LeakyReLU(),
                          nn.Linear(hidden_dim,hidden_dim),
                          nn.LeakyReLU(),
                          nn.Linear(hidden_dim,output_dim))

#discriminator
discriminator = nn.Sequential(nn.Linear(output_dim,hidden_dim),
                          nn.ReLU(),
                          nn.Linear(hidden_dim,hidden_dim),
                          nn.ReLU(),
                          nn.Linear(hidden_dim,1),
                          nn.Sigmoid())

#loss func
loss = nn.BCELoss()
optimizer_g = optim.Adam(generator.parameters(),lr=learning_rate)
optimizer_d = optim.Adam(discriminator.parameters(),lr=learning_rate)

#training the gen and dis
#generating the data real and fake

for epoch in range(num_epochs+1):
    real_data = torch.normal(mean=0,std=1,size=(batch_size,1))
    real_labels = torch.ones(size=(batch_size,1))

    #generating fake numbers
    random_samples = torch.randn(batch_size,input_dim)
    fake_data = generator(random_samples)
    fake_labels = torch.zeros(size=(batch_size,1))

    #now training the geerator
    optimizer_g.zero_grad()
    generated_data = generator(random_samples)
    generated_labels = discriminator(generated_data)
    loss_g = loss(generated_labels,real_labels)
    loss_g.backward()
    optimizer_g.step()

    #training the discriminator
    optimizer_d.zero_grad()
    real_pred = discriminator(real_data)
    real_loss = loss(real_pred,real_labels)
    fake_pred = discriminator(fake_data.detach())
    fake_loss = loss(fake_pred,fake_labels)
    total_loss = real_loss+fake_loss
    total_loss.backward()
    optimizer_d.step()

    if epoch % 500 == 0:
        print(f"Epoch : {epoch} || Loss : {total_loss.item()}")
        print("--------------------------------------------------")


#for plot demonstration
real_samples = torch.normal(mean=0,std=1,size=(1000,1)).numpy()
generated_samples = generator(torch.randn(1000,input_dim)).detach().numpy()

plt.figure(figsize=(10, 5))
plt.hist(real_samples,bins=30,label="Real_data")
plt.hist(generated_samples,bins=30,label="Generated_data")
plt.legend()
plt.title("Real vs Generated data")
plt.show()
