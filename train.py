import numpy as np
from bm import BMNet
from bmimage import BMImage
#from keras.datasets import mnist
import matplotlib.pyplot as plt
from dbnet import DBNet
from dbimage import DBImage
plt.rcParams['image.cmap'] = 'Greys_r'

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
    train_X = np.load(f)
with open(path+'data/train_labels.npy', 'rb') as f:
    train_labels = np.load(f)
with open(path+'data/test_data.npy', 'rb') as f:
    test_X = np.load(f)
with open(path+'data/test_labels.npy', 'rb') as f:
    test_labels = np.load(f)

print("Done loading MNIST")

n_test = 10000
test_X = test_X[:n_test,:,:]
test_labels = test_labels[:n_test]

model = DBImage(28,28,10,[500,500,2000],learning_rate=0.01,momentum=0.5,weight_decay=10**-5)
model.set_images(train_X, train_labels,split=1)

model.load_weights(path+"models/db_")

model.validate(test_X,test_labels,full=True,filename=path+"figures/DBN/")
model.train(100,save=True, filename=path + "models/db_")
#model.validate(test_X,test_labels,filename=path+"figures/", full=False)
#model.tune(1)
model.validate(test_X,test_labels,full=True,filename=path+"figures/DBN/")

for i in range(10):
    print("Generating: ", str(i))
    lb = np.zeros(10)
    lb[i] = 1
    model.generate(np.array(lb),filename="figures/DBN/gen"+str(i))
