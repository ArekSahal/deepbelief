import numpy as np
import torch
#from keras.datasets import mnist
import matplotlib.pyplot as plt
from dbimage_torch import DBImage
plt.rcParams['image.cmap'] = 'Greys_r'
path=""

with open(path+'data/train_data.pt', 'rb') as f:
    train_X = torch.load(f)
with open(path+'data/train_labels.pt', 'rb') as f:
    train_labels = torch.load(f)
with open(path+'data/test_data.pt', 'rb') as f:
    test_X = torch.load(f)
with open(path+'data/test_labels.pt', 'rb') as f:
    test_labels = torch.load(f)

print("Done loading MNIST")

n_test = 10000
n_train = 60000
test_X = test_X[:n_test,:,:].float()
train_X = train_X[:n_train,:,:].float()
test_labels = test_labels[:n_test].float()
train_labels = train_labels[:n_train].float()

model = DBImage(28,28,10,[500,500,2000],
path_writer="runs/WTF")
model.set_images(train_X, train_labels,split=1)

model.load_weights(path+"models/db_tuned_")

x = 35
#model.validate(test_X,test_labels,full=True,generate=True, no_print=False)
model.daydream(test_X[x,:]/255., test_labels[x,:])