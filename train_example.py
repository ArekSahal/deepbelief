import torch
from keras.datasets import mnist
import matplotlib.pyplot as plt
from dbimage import DBImage

path = ""

(train_X, train_labels), (test_X, test_labels) = mnist.load_data()

n_test = 10000
n_train = 10000
test_X = test_X[:n_test,:,:].float()
train_X = train_X[:n_train,:,:].float()
test_labels = test_labels[:n_test].float()
train_labels = train_labels[:n_train].float()

model = DBImage(28,28,10,[500,500,2000],
batch_size=20,
learning_rate=0.01,
momentum=0.9,
initial_momentum=0.5,
weight_decay=2*10**-6,
tune_rate=0.001,
tune_decay=0.0001,
tune_momentum=0.5,
path_writer="runs/Test/tuned",
sparsity_target=0.01,
sparsity_penalty=0.0)

model.set_images(train_X, train_labels,split=0.7)

model.load_weights(path+"models/db_tuned_")

model.validate(test_X,test_labels,full=True,filename=path+"figures/DBN/")
#model.train(1,save=False, filename=path + "models/db_")
print(model.predict(test_X[:2,:]))
print(test_labels[:2,:])
#model.validate(test_X,test_labels,filename=path+"figures/", full=True)
#model.tune(20,save=False)
#model.validate(test_X,test_labels,full=True,filename=path+"figures/DBN/")
