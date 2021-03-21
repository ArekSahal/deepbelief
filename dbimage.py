from dbnet import DBNet
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
    path_writer="runs/DBN",
    tb=False,
    lb_names=False,
    units=[]):
        self.lb_names = lb_names
        self.writer = SummaryWriter(path_writer)
        self.n_visible = height*width
        self.width = width
        self.height = height
        self.n_labels = n_labels

        self.model = DBNet(self.n_visible,hidden,self.n_labels,batch_size=batch_size,learning_rate=learning_rate,momentum=momentum,initial_momentum=initial_momentum,weight_decay=weight_decay,tune_rate=tune_rate, tune_decay=tune_decay, tune_momentum=tune_momentum,cdns=cdns,sparsity_penalty=sparsity_penalty,sparsity_target=sparsity_target, path_writer=path_writer, tb=tb, units=units)

        self.model.stack[0].tb_funcs.append(self.show_weights)
        self.flat_images = []
        self.n_labels = []

    def set_images(self, images, labels,split=0.8):
        #self.images = (images - torch.mean(images))/torch.std(images)
        self.images = images/255.0
        self.flat_images = []
        for i in range(images.shape[0]):
            self.flat_images.append(self.images[i,:,:].flatten())
        self.model.set_training_data(torch.stack(self.flat_images), labels,split=split)

    def train(self,epochs,save=True, filename="models/db",test=False, full_loss=False):
        self.model.train(epochs,save, filename=filename,test=test,full_loss=full_loss)
    def predict(self,X):
        data = []
        for i in range(X.shape[0]):
            data.append(X[i,:,:].flatten())
        #data = (torch.stack(data) - torch.mean(torch.stack(data)))/torch.std(torch.stack(data))
        data = torch.stack(data)/255.0
        return self.model.predict(data)



    def validate(self, test_data=False, test_labels=False, full=True, filename="figures/DBN/", no_print=False, generate=True, to_tb=False, to_file=False):
        print("Validating model")
        count = 0
        if torch.is_tensor(test_data):
            data = []
            for i in range(test_data.shape[0]):
                data.append(test_data[i,:,:].flatten())
            #data = (torch.stack(data) - torch.mean(torch.stack(data)))/torch.std(torch.stack(data))
            data = torch.stack(data)/255.0
            count, preds = self.model.validate(data,test_labels)

            if not no_print:
                print("Test accuracy: ", count)
            if full:
                mess_img = []
                mess_pred = []
                mess_true =[]

                confusion_list = [[0,0,0,0,0,0,0,0,0,0] for i in range(10)]

                for i in range(test_labels.shape[0]):
                    pred = torch.argmax(preds[i]).to("cpu")
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
                if self.lb_names:
                    ax.set_xticklabels(self.lb_names)
                    ax.set_yticklabels(self.lb_names)
                else:
                    ax.set_xticklabels(torch.arange(10).tolist())
                    ax.set_yticklabels(torch.arange(10).tolist())
                plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

                ax.set_xlabel("Label")
                ax.set_ylabel("Prediction")
                ax.set_title("Confusion matrix")

                ax.imshow(confusion_matrix.t(), cmap="magma")

                for i in range(10):
                    for j in range(10):
                        text = ax.text(j, i, torch.round(100*confusion_matrix[j][i]).tolist() /100,
                                    ha="center", va="center", color="w")
                plt.tight_layout()

                self.writer.add_figure("confusion matrix", fig)

                fig, axs = plt.subplots(3,3)
                for i in range(3):
                    for j in range(3):
                        axs[i][j].imshow(mess_img[i*3 + j].reshape(self.height,self.width), cmap="Greys_r")
                        if self.lb_names:
                            axs[i][j].text(2,3,self.lb_names[int(mess_pred[i*3 + j].tolist())], color="r")
                        else:
                            axs[i][j].text(2,3,mess_pred[i*3 + j].tolist(), color="r")
                        axs[i][j].get_xaxis().set_visible(False)
                        axs[i][j].get_yaxis().set_visible(False)

                self.writer.add_figure("wrong predictions", fig)
                self.show_weights(to_tb=True)
        else:
            count = self.model.validate()
        
        if generate:
            for i in range(10):
                if self.lb_names:
                    nm = self.lb_names[i]
                else:
                    nm = i
                print("Generating: ", str(nm))
                lb = torch.zeros(10)
                lb[i] = 1.
                self.generate(lb,filename=str(nm), to_tb=to_tb, to_file=to_file)
        
        return count

    def generate(self, label,n=1000,to_file=False, to_tb=False, filename="gen"):
        lb= label
        iters = self.model.generate(lb,n=n)
        if to_tb or to_file:
            video = torch.zeros(1,len(iters),3,self.height,self.width)
            for i in range(len(iters)):
                dummy = (iters[i] - torch.min(iters[i])).reshape(self.height,self.width)
                dummy = dummy / torch.max(dummy)
                video[0,i,:,:,:] = dummy.float()
            if to_tb:
                self.writer.add_video("gen" + str(filename), video,fps=12)
            if to_file:
                torchvision.io.write_video(filename + ".mp4", 255*video[0,:,:,:,:].permute(0,2,3,1), fps=12)
                
        return iters
    
    def daydream(self,X, label,to_file=False, to_tb=True, filename="dreamX"):
        lb= label
        data = torch.stack([X.flatten()])
        iters = self.model.daydream(data,lb,n=1000)
        if to_file or to_tb:
            video = torch.zeros(1,len(iters),3,self.height,self.width)
            for i in range(len(iters)):
                video[0,i,:,:,:] = iters[i].reshape(self.height,self.width)
            if to_file:
                torchvision.io.write_video(filename +".mp4", 255*video[0,:,:,:,:].permute(0,2,3,1), fps=12)
            if to_tb:
                self.writer.add_video(str(filename), video, fps=12)
        return iters
    

    def load_weights(self,filename="models/db_"):
        self.model.load_weights(filename)

    def tune(self,epochs=[20],save=True,cd=[3],filename="models/db_"):
        for ep in range(len(epochs)):
            self.model.wakesleep_finetune(epochs[ep],cd=cd[ep],save=save, filename=filename)

    def show_weights(self,n=10,step=0,eps=0, to_file=False, to_tb=False, filename="weights"):
        images = torch.zeros((n*n,3,self.height,self.width))
        for i in range(n*n):
            dummy = self.model.stack[0].w[:,i].reshape(self.height,self.width) - torch.min(self.model.stack[0].w[:,i])
            images[i,:,:,:] = dummy/torch.max(dummy)
        img_grid = torchvision.utils.make_grid(images,nrow=n)
        if to_tb:
            self.writer.add_image("Hidden weights",img_grid,global_step=step)
        if to_file:
            torchvision.io.write_png(img_grid, filename)
        return img_grid


