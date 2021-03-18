
from keras.datasets import fashion_mnist
import torch
import matplotlib.pyplot as plt

path=""
(train_X, train_y), (test_X, test_y) = fashion_mnist.load_data()
train_labels = []
for i in range(train_y.shape[0]):
    lb = torch.zeros(10)
    lb[train_y[i]] = 1
    train_labels.append(lb)
train_labels = torch.stack(train_labels)

test_labels = []
for i in range(test_y.shape[0]):
    lb = torch.zeros(10)
    lb[test_y[i]] = 1
    test_labels.append(lb)

test_labels = torch.stack(test_labels)

train_X = torch.from_numpy(train_X)
test_X = torch.from_numpy(test_X)

with open('data/train_data_f.pt', 'wb') as f:
    torch.save(train_X,f)
with open('data/train_labels_f.pt', 'wb') as f:
    torch.save( train_labels,f)
with open('data/test_data_f.pt', 'wb') as f:
    torch.save(test_X,f)
with open('data/test_labels_f.pt', 'wb') as f:
    torch.save(test_labels,f)


with open(path+'data/train_data_f.pt', 'rb') as f:
    train_X = torch.load(f)
with open(path+'data/train_labels_f.pt', 'rb') as f:
    train_labels = torch.load(f)
with open(path+'data/test_data_f.pt', 'rb') as f:
    test_X = torch.load(f)
with open(path+'data/test_labels_f.pt', 'rb') as f:
    test_labels = torch.load(f)

print(train_labels[0,:])
plt.imshow(train_X[0,:,:])
plt.show()
print("Done loading FASHION_MNIST")