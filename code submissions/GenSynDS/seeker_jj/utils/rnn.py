"""
Minimal RNN.
"""
import torch
from torch import nn
import numpy as np


SEQ_LEN = 5
INPUT_SIZE = 23
BATCH = 17
N_STATES = 26
LEARNING_RATE = 1e-1
BPTT = 4


class RNN(nn.Module):
    '''
    Should take a tensor of integers of size SEQ_LEN x BATCH
    '''
    def __init__(self, input_size, hidden_size, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(N_STATES, input_size)
        # Input of LSTM has shape (SEQ_LEN, BATCH, INPUT_SIZE)
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers)
        self.linear = nn.Linear(hidden_size, N_STATES)

    def forward(self, x):
        x = self.embedding(x)
        latent, _ = self.lstm(x)
        logits = self.linear(latent)
        return logits


model = RNN(input_size=INPUT_SIZE, hidden_size=12)


def tok2num(c):
    '''
    For the minimal example, converting letters to integers.
    '''
    return ord(c) - ord('a')


# Prepare data
X = []
y = []
corpus = ['pokemon', 'anticonstitutionnellement', 'antipasti', 'antidote', 'pokebowl', 'pokeball']
for line in corpus:
    X.append(list(map(tok2num, line[:BPTT])))
    y.append(list(map(tok2num, line[1:BPTT + 1])))
X = torch.from_numpy(np.array(X).T)
y = torch.from_numpy(np.array(y).T).reshape(-1)
print(X.T)
print(y.T)

cross_entropy = nn.CrossEntropyLoss()
batch = X  # Overfit single batch

losses = []
opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
for _ in range(10):
    pred = model(batch).reshape(-1, N_STATES)
    print(pred.argmax(-1)[:5], y[:5])
    loss = cross_entropy(pred, y)
    opt.zero_grad()
    loss.backward()
    opt.step()
    losses.append(loss)
    print('loss', loss)
