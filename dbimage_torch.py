from dbnet_torch import DBNet
import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from torch.utils.tensorboard import SummaryWriter
if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"
    
class DBImage:

    def __init__(self, height,width,n_labels,hidden,
    batch_size=20,
    learning_rate=0.05,
    momentum=0.7,
    initial_momentum=0.5,
    weight_decay=0.0001,
    tune_rate=0.05, 
    tune_momentum=0.5, 
    tune_decay=0.07, 
    cdns=False,
    sparsity_penalty=0.05,
    sparsity_target=0.1,
    path_writer="runs/DBN"):

        self.writer = SummaryWriter(path_writer)
        self.n_visible = height*width
        self.width = width
        self.height = height
        self.n_labels = n_labels

        self.model = DBNet(self.n_visible,hidden,self.n_labels,batch_size=batch_size,learning_rate=learning_rate,momentum=momentum,initial_momentum=initial_momentum,weight_decay=weight_decay,tune_rate=tune_rate, tune_decay=tune_decay, tune_momentum=tune_momentum,cdns=cdns,sparsity_penalty=sparsity_penalty,sparsity_target=sparsity_target, path_writer=path_writer)

        #self.model.stack[0].tb_funcs.append(self.show_weights)
        self.flat_images = []
        self.n_labels = []

    def set_images(self, images, labels,split=0.8):
        self.images = images/255.0
        self.flat_images = []
        for i in range(images.shape[0]):
            self.flat_images.append(self.images[i,:,:].flatten())
        self.model.set_training_data(torch.stack(self.flat_images), labels,split=split)

    def train(self,epochs,save=True, filename="models/db",test=False, full_loss=False):
        self.model.train(epochs,save, filename=filename,test=test,full_loss=full_loss)

    def validate(self, test_data, test_labels, full=True, filename="figures/DBN/", no_print=False):
        print("Validating model")
        data = []
        for i in range(test_data.shape[0]):
            data.append(test_data[i,:,:].flatten())
        data = torch.stack(data)/255.0
        preds_dirty = self.model.predict(data)
        preds = torch.zeros(preds_dirty.shape)
        for i in range(preds_dirty.shape[0]):
            preds[i,torch.argmax(preds_dirty[i,:])] = 1
        count = 0
        for i in range(test_labels.shape[0]):
            if torch.all(test_labels[i,:] == preds[i,:]):
                count += 1
        if not no_print:
            print("Test accuracy: ", count / test_labels.shape[0])
        if full:
            mess_img = []
            mess_pred = []
            mess_true =[]

            confusion_list = [[0,0,0,0,0,0,0,0,0,0] for i in range(10)]

            for i in range(test_labels.shape[0]):
                pred = torch.argmax(preds[i])
                l = torch.argmax(test_labels[i])
                confusion_list[l][pred] = confusion_list[l][pred] + 1
                if l != pred:
                    mess_img.append(test_data[i:i+1,:])
                    mess_pred.append(pred)
                    mess_true.append(l)

            print("Confusion Matrix")
            confusion_matrix = torch.tensor(confusion_list)
            confusion_matrix = confusion_matrix/torch.sum(confusion_matrix,dim=1)
            fig, ax = plt.subplots()
            ax.set_xticks(torch.arange(10).tolist())
            ax.set_yticks(torch.arange(10).tolist())
            ax.set_xticklabels(torch.arange(10).tolist())
            ax.set_yticklabels(torch.arange(10).tolist())

            ax.set_xlabel("Label")
            ax.set_ylabel("Prediction")
            ax.set_title("Confusion matrix")

            ax.imshow(confusion_matrix.t(), cmap="viridis")

            for i in range(10):
                for j in range(10):
                    text = ax.text(j, i, torch.round(100*confusion_matrix[j][i]).tolist() /100,
                                ha="center", va="center", color="w")

            self.writer.add_figure("confusion matrix", fig)

            fig, axs = plt.subplots(3,3)
            for i in range(3):
                for j in range(3):
                    axs[i][j].imshow(mess_img[i*3 + j].reshape(self.height,self.width), cmap="Greys_r")
                    axs[i][j].text(2,3,mess_pred[i*3 + j].tolist(), color="w")
                    axs[i][j].get_xaxis().set_visible(False)
                    axs[i][j].get_yaxis().set_visible(False)
            self.show_weights()

            self.writer.add_figure("wrong predictions", fig)
            for i in range(10):
                print("Generating: ", str(i))
                lb = torch.zeros(10)
                lb[i] = 1.
                self.generate(lb,filename=str(i))
        
        return count / float(test_labels.shape[0])

    def generate(self, label,filename="gen"):
        lb= label
        iters = self.model.generate(lb,n=1000)
        video = torch.zeros(1,len(iters),3,self.height,self.width)
        for i in range(len(iters)):
            video[0,i,:,:,:] = iters[i].reshape(self.height,self.width)
        self.writer.add_video("gen" + str(filename), video,fps=10)

        

    def load_weights(self,filename="models/db_"):
        self.model.load_weights(filename)

    def tune(self,epochs,save=True,filename="models/db_"):
        self.model.wakesleep_finetune(epochs,save=save, filename=filename)
    
    def show_losses(self,filename="figures/DBN/"):
        self.model.show_losses(filename)

    def show_weights(self,n=10,step=0,eps=0):
        images = torch.zeros((n*n,3,self.height,self.width))
        for i in range(n*n):
            dummy = self.model.stack[0].w[:,i].reshape(self.height,self.width) - torch.min(self.model.stack[0].w[:,i])
            images[i,:,:,:] = dummy/torch.max(dummy)
        img_grid = torchvision.utils.make_grid(images,nrow=n)
        
        self.writer.add_image("hidden weights",img_grid,global_step=step)


