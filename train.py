# _*_ coding: utf-8 _*_
import numpy as np
import random
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from nltk_utils import bag_of_words, tokenize, stem
from model import NeuralNet

with open('intents.json', 'r', encoding="utf-8") as f:
    intents = json.load(f)

all_words = []
tags = []
xy = []

# loop através de cada sentença nos padrões de intenções
for intent in intents['intents']:
    tag = intent['tag']
    # adiciona à lista de tags
    tags.append(tag)
    for pattern in intent['patterns']:
        # tokeniza cada palavra na sentença
        w = tokenize(pattern)
        # adiciona à nossa lista de palavras
        all_words.extend(w)
        # adiciona ao par xy
        xy.append((w, tag))

# stem e transforma em minúsculas cada palavra
ignore_words = ['?', '.', '!']
all_words = [stem(w) for w in all_words if w not in ignore_words]
# remove duplicatas e ordena
all_words = sorted(set(all_words))
tags = sorted(set(tags))

print(len(xy), "padrões")
print(len(tags), "tags:", tags)
print(len(all_words), "palavras únicas stemizadas:", all_words)

# cria os dados de treinamento
X_train = []
y_train = []
for (pattern_sentence, tag) in xy:
    # X: bag of words para cada pattern_sentence
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)
    # y: CrossEntropyLoss do PyTorch precisa apenas de rótulos de classe, não one-hot
    label = tags.index(tag)
    y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)

# Hiperparâmetros
num_epochs = 1000
batch_size = 8
learning_rate = 0.001
input_size = len(X_train[0])
hidden_size = 8
output_size = len(tags)
print(input_size, output_size)


class ChatDataset(Dataset):

    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    # suporta indexação para que dataset[i] possa ser usado para obter a i-ésima amostra
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    # podemos chamar len(dataset) para retornar o tamanho
    def __len__(self):
        return self.n_samples


dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = NeuralNet(input_size, hidden_size, output_size).to(device)

# Função de perda e otimizador
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Treina o modelo
for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)

        # Passagem direta
        outputs = model(words)
        # se y fosse one-hot, deveríamos aplicar
        # labels = torch.max(labels, 1)[1]
        loss = criterion(outputs, labels)

        # Retropropagação e otimização
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'Época [{epoch + 1}/{num_epochs}], Perda: {loss.item():.4f}')

print(f'perda final: {loss.item():.4f}')

data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "hidden_size": hidden_size,
    "output_size": output_size,
    "all_words": all_words,
    "tags": tags
}

FILE = "data.pth"
torch.save(data, FILE)

print(f'treinamento completo. arquivo salvo em {FILE}')
