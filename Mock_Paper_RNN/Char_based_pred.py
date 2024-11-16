import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import spacy
import string

# Load spaCy's English tokenizer and vocabulary
nlp = spacy.blank("en")

# Step 1: Create a character-level vocabulary
characters = list(string.digits + string.punctuation + ' ')
vocab = sorted(set(characters + [char for word in nlp.vocab if word.is_alpha for char in word.text.lower()]))
vocab_size = len(vocab)

# Character to index mappings
char_to_idx = {ch: i for i, ch in enumerate(vocab)}
idx_to_char = {i: ch for i, ch in enumerate(vocab)}

# Load text data
with open('pizza.txt', 'r', encoding='utf-8') as file:  # Ensure utf-8 for compatibility with special characters
    text_data = file.read()

sequence_length = 100
encoded_text = [char_to_idx.get(c, 0) for c in text_data]  # Encode text with indices
dataset_length = len(encoded_text) - sequence_length

# Prepare dataset
sequences = [(encoded_text[i:i + sequence_length], encoded_text[i + 1:i + sequence_length + 1]) for i in range(dataset_length)]
inputs, targets = zip(*sequences)
inputs = torch.tensor(inputs)
targets = torch.tensor(targets)
dataloader = DataLoader(list(zip(inputs, targets)), batch_size=32, shuffle=True)

# Define model
embedding_dim = 64
hidden_dim = 128
n_layers = 2
embedding = nn.Embedding(vocab_size, embedding_dim)
lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, batch_first=True)
fc = nn.Linear(hidden_dim, vocab_size)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(list(embedding.parameters()) + list(lstm.parameters()) + list(fc.parameters()), lr=0.001)

# Training loop
n_epochs = 50
for epoch in range(n_epochs):
    total_loss = 0
    for batch_inputs, batch_targets in dataloader:
        optimizer.zero_grad()
        batch_size = batch_inputs.size(0)

        # Initialize hidden state
        hidden = (
            torch.zeros(n_layers, batch_size, hidden_dim),
            torch.zeros(n_layers, batch_size, hidden_dim)
        )

        # Forward pass
        embedded = embedding(batch_inputs)
        output, hidden = lstm(embedded, hidden)
        output = fc(output)

        # Reshape for loss calculation
        output = output.view(-1, vocab_size)
        batch_targets = batch_targets.view(-1)

        loss = criterion(output, batch_targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch + 1}, Loss: {total_loss / len(dataloader):.4f}")

# Save the model parameters
torch.save({
    'embedding_state_dict': embedding.state_dict(),
    'lstm_state_dict': lstm.state_dict(),
    'fc_state_dict': fc.state_dict(),
    'vocab': vocab,
    'char_to_idx': char_to_idx,
    'idx_to_char': idx_to_char
}, "char_lstm_model.pth")

# Sampling to generate text
start_text = "temperature"
length = 280
model_data = torch.load("char_lstm_model.pth")

# Reload saved parameters
embedding.load_state_dict(model_data['embedding_state_dict'])
lstm.load_state_dict(model_data['lstm_state_dict'])
fc.load_state_dict(model_data['fc_state_dict'])

# Generate text
chars = [char_to_idx[c] for c in start_text.lower()]
hidden = None
for _ in range(length):
    x = torch.tensor([chars[-1]]).unsqueeze(0)
    embedded = embedding(x)
    output, hidden = lstm(embedded, hidden)
    last_char = fc(output).argmax(dim=2).item()
    chars.append(last_char)

generated_text = ''.join(idx_to_char[idx] for idx in chars)
print("Generated text:", generated_text)
