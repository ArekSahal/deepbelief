import numpy as np
import torch
#from keras.datasets import mnist
import matplotlib.pyplot as plt
from dbimage_torch import DBImage
plt.rcParams['image.cmap'] = 'Greys_r'
path=""

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

model = DBImage(28,28,10,[500,500,2000],learning_rate=0.01,momentum=0.7,weight_decay=10**-5)
model.set_images(train_X, train_labels,split=1)

model.load_weights(path+"models/db_")

model.validate(test_X,test_labels,full=True,filename=path+"figures/DBN/")
model.show_losses(path+"figures/DBN/")

"""
for i in range(10):
    print("Generating: ", str(i))
    lb = torch.zeros(10)
    lb[i] = 1
    model.generate(lb,filename="figures/DBN/gen"+str(i))

"""