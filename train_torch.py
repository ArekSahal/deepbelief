import numpy as np
import torch
#from keras.datasets import mnist
import matplotlib.pyplot as plt
from dbimage_torch import DBImage
from datetime import datetime
plt.rcParams['image.cmap'] = 'Greys'

path = ""


print("Loading MNIST dataset")
"""
(train_X, train_y), (test_X, test_y) = mnist.load_data()
train_labels = []
for i in range(train_y.shape[0]):
    lb = np.zeros(10)
    lb[train_y[i]] = 1
    train_labels.append(lb)
train_labels = np.array(train_labels)

test_labels = []
for i in range(test_y.shape[0]):
    lb = np.zeros(10)
    lb[test_y[i]] = 1
    test_labels.append(lb)

test_labels = np.array(test_labels)

with open('data/train_data.npy', 'wb') as f:
    np.save(f, train_X)
with open('data/train_labels.npy', 'wb') as f:
    np.save(f, train_labels)
with open('data/test_data.npy', 'wb') as f:
    np.save(f, test_X)
with open('data/test_labels.npy', 'wb') as f:
    np.save(f, test_labels)


"""
with open(path+'data/train_data.npy', 'rb') as f:
    train_X = torch.from_numpy(np.load(f))
with open(path+'data/train_labels.npy', 'rb') as f:
    train_labels = torch.from_numpy(np.load(f))
with open(path+'data/test_data.npy', 'rb') as f:
    test_X = torch.from_numpy(np.load(f))
with open(path+'data/test_labels.npy', 'rb') as f:
    test_labels = torch.from_numpy(np.load(f))


print("Done loading MNIST")

n_test = 10000
n_train = 60000
test_X = test_X[:n_test,:,:].float()
train_X = train_X[:n_train,:,:].float()
test_labels = test_labels[:n_test].float()
train_labels = train_labels[:n_train].float()

model = DBImage(28,28,10,[500,500,2000],
batch_size=20,
learning_rate=0.01,
momentum=0.1,
initial_momentum=0.1,
weight_decay=2*10**-4,
tune_rate=0.001,
path_writer="runs/MNIST/" + str(datetime.now()) + "_DBN",
sparsity_target=0.01,
sparsity_penalty=0.00001)

model.set_images(train_X, train_labels,split=0.9)

#model.load_weights(path+"models/db_")

#model.validate(test_X,test_labels,full=True,filename=path+"figures/DBN/")
model.train(20,save=True, filename=path + "models/db_")
model.validate(test_X,test_labels,filename=path+"figures/", full=True)
#model.tune(5)
#model.validate(test_X,test_labels,full=True,filename=path+"figures/DBN/")
