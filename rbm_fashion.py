import numpy as np
import torch
import torchvision
from bm_torch import BMNet
from bmimage_torch import BMImage
#from keras.datasets import mnist
import matplotlib.pyplot as plt
plt.rcParams['image.cmap'] = 'Greys_r'
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

path = ""

with open(path+'data/train_data_f.pt', 'rb') as f:
    train_X = torch.load(f)
with open(path+'data/train_labels_f.pt', 'rb') as f:
    train_labels = torch.load(f)
with open(path+'data/test_data_f.pt', 'rb') as f:
    test_X = torch.load(f)
with open(path+'data/test_labels_f.pt', 'rb') as f:
    test_labels = torch.load(f)

train_X_index = torch.tensor(np.random.choice(60000,10000))
test_X_index = torch.tensor(np.random.choice(10000,100))

train_X = train_X[train_X_index,:,:]
lbs = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle Boot"]

"""
target = 5
train_X = train_X[torch.argmax(train_labels,dim=1) == target,:]
train_labels = train_labels[torch.argmax(train_labels,dim=1) == target,:]
"""

model = BMImage(28,28,
    hidden=200,
    batch_size=100,
    learning_rate=0.1,
    momentum=0.9,
    weight_decay=2*10**-3,
    sparsity_penalty=0.1,
    sparsity_target=0.01,
    cdn=1,
    initial_momentum=0.5,
    path_writer=path+'runs/FNIST/RBM/10')

model.set_images(train_X,split=0.95)
model.train(100,save=True,filename="models/rbm_")
model.reconstruct()
#model.load_weights("models/db_0")
#model.load_weights("models/rbm_")
#model.load_weights("models/db_")
#test_image = torch.rand(28,28)
#test_image = test_X[np.random.randint(0,test_X.shape[0]),:,:]/255.0 



#ax.fill_between(torch.arange(len(model.model.mean_delta_ws)),torch.tensor(model.model.mean_delta_ws) - torch.tensor(model.model.std_delta_ws),torch.tensor(model.model.mean_delta_ws) + torch.tensor(model.model.std_delta_ws),alpha=0.5 )
#fig.savefig(path+"figures/RBM/datla_ws.png", dpi=600)
#plt.figure()
#print(model.model.w)
#plt.hist(model.model.w.flatten())
#plt.savefig("figures/RBM/weights_hist.png",dpi=600)
#plt.tight_layout()
#plt.show()

