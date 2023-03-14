import numpy as np
import torch
from torch.utils import data
from torch import nn
from d2l import torch as d2l

true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 1000)

def load_array(data_arrays, batch_size, is_train=True): #@save
	dataset = data.TensorDataset(*data_arrays)
	return data.DataLoader(dataset, batch_size, shuffle=is_train)

batch_size = 10
data_iter = load_array((features, labels), batch_size)

# The model we want to get.
net = nn.Sequential(nn.Linear(2, 1))
net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)

# loss function, MSELoss is just an L2 norm.
loss = nn.MSELoss()

# The trainer to tune the parameters in the model.
# lr is the learning rate.
trainer = torch.optim.SGD(net.parameters(), lr=0.03)

num_epochs = 3
for epoch in range(num_epochs):
	for X, y in data_iter:
		l = loss(net(X), y)
		# clear the gradient from the last iteration.
		trainer.zero_grad()
		# get the gradient for the current loss.
		l.backward()
		# revalue net.parameters() according to the current gradient.
		# if net(X) > y, we need to reduce the parameters and vice versa.
		trainer.step()
	l = loss(net(features), labels)
	print(f'epoch {epoch + 1}, loss {l:f}')

w = net[0].weight.data
b = net[0].bias.data
print (f'w: {w}, b: {b}')