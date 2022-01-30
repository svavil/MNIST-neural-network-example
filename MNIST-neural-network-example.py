import pandas as pd
import numpy as np
import torch
from torch import nn
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import matplotlib.pyplot as plt

train = pd.read_csv("Downloads/train.csv.zip")

# convert Kaggle data to PyTorch tensors
y_train = train.label.values.reshape(-1, 1)
enc = OneHotEncoder(sparse = False).fit(y_train)
y_train = torch.tensor(enc.transform(y_train), dtype = torch.float)
X_train = torch.tensor(train.drop(labels = "label", axis = "columns").values, dtype = torch.float)

# define a network with two layers: 784 input neurons, 256 hidden neurons and 10 output neurons
class FCN(nn.Module):
    def __init__(self):
        super(FCN, self).__init__()
        self.flatten = nn.Flatten()
        self.model1 = nn.Linear(28*28, 256)
        self.model2 = nn.Linear(256, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.flatten(x)
        logits = self.model2(self.relu(self.model1(x)))
        return logits
    
# check that the model works
fcn = FCN()
predictions = fcn(X_train)
assert(predictions.shape == y_train.shape)

# metrics before learning
predicted_classes = np.argmax(predictions.detach().numpy(), axis = 1)
print(confusion_matrix(predicted_classes, train.label.values))

plt.imshow(fcn.model1.weight.detach().numpy(), extent = (0, 10, 0, 10), aspect = 1, interpolation = "nearest")
plt.colorbar()
plt.show()

# learning parameters
learn_rate = 1e-3
epochs = 1000
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(fcn.parameters(), lr = learn_rate)
pbar = tqdm(range(epochs))

# learning loop
for e in pbar:
    predictions = fcn(X_train)
    assert(predictions.shape == y_train.shape)
    loss = loss_fn(predictions, y_train)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    pbar.set_postfix({"Loss": round(loss.item(), 2)})
print("")

# metrics after learning
predicted_classes = np.argmax(predictions.detach().numpy(), axis = 1)
print(confusion_matrix(predicted_classes, train.label.values))

for name, param in fcn.named_parameters():
    print(f"{name} is a tensor with shape {param.shape}")

plt.imshow(fcn.model2.weight.detach().numpy(), extent = (0, 10, 0, 10), aspect = 1, interpolation = "nearest")
plt.colorbar()
plt.show()