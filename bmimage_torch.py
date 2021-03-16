import numpy as np
import torch
from bm_torch import BMNet
import matplotlib.pyplot as plt
import matplotlib.animation as ani
from torch.utils.tensorboard import SummaryWriter
import torchvision

if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"
    

class BMImage:

    def __init__(self, width=100, height=100, hidden=100,
    batch_size=20,
    learning_rate=0.01,
    momentum=0.7,
    weight_decay=10**-4,
    sparsity_penalty=0.05,
    sparsity_target=0.1,
    cdn=1,
    initial_momentum=0.5,
    path_writer=False):
        
        self.tb_funcs = [self.show_weights]
        self.width = width
        self.height = height
        self.hidden = hidden
        self.writer=SummaryWriter(path_writer)

        self.model = BMNet(width*height, 
        hidden,
        batch_size=batch_size,
        learning_rate=learning_rate,
        momentum=momentum,
        weight_decay=weight_decay, 
        sparsity_penalty=sparsity_penalty,
        sparsity_target=sparsity_target,
        cdn=cdn,
        initial_momentum=initial_momentum,
        tb_funcs=self.tb_funcs,
        writer=self.writer)

        self.images = []
        self.flat_images = []
        self.flat_val_images = []



    def set_images(self, images,split=1):

        self.images = images/255.0

        self.flat_images = []
        for i in range(images.shape[0]):
            self.flat_images.append(self.images[i,:,:].flatten())

        data = torch.stack(self.flat_images)
        if split>=1.0:
            self.flat_images = data
        else:
            shuffl_index = np.arange(data.shape[0])
            np.random.shuffle(shuffl_index)
            data = data[shuffl_index,:]

            lmt = int(data.shape[0]*split)

            self.flat_images = data[:lmt,:]

            self.flat_val_images = data[lmt:,:]
        
        if len(self.flat_val_images)>0:
            self.model.set_data(self.flat_images,self.flat_val_images)
        else:
            self.model.set_data(torch.stack(self.flat_images))

    def train(self, epochs, show=True, save=False,filename="models/rbm",full_loss=True):
        loss = self.model.train(epochs, save=save,full_loss=full_loss, filename=filename)
        if save:
            self.model.save_weights(filename=filename)
        self.reconstruct()
        
        return loss
    
    def load_weights(self, filename="models/model"):
        self.model.load_weights(filename)

    def reconstruct(self, n=100):
        im = self.flat_val_images[0,:].to(device)
        video = torch.zeros(1,n+1,3,self.height,self.width).to(device)
        flat_im = im
        video[0,0,:,:,:] = self.model.reconstruct(torch.stack([flat_im]).float(),n=1).reshape(self.height,self.width)
        for i in range(n):
            video[0,i,:,:,:]  = self.model.reconstruct(torch.stack([video[0,-1,0,:,:].flatten()]),n=1).reshape(self.height,self.width)
        self.writer.add_video("reconstruction",video)




    def show_weights(self,n=10,filename="figures/RBM/",step=0,eps=0):
        if n*n > self.model.n_hidden:
            n=int(torch.sqrt(torch.tensor(self.model.n_hidden)).tolist())
        images = torch.zeros((n*n,3,self.height,self.width)).to(device)
        for i in range(n*n):
            dummy = self.model.w[:,i].reshape(self.height,self.width) - torch.min(self.model.w[:,i])
            images[i,:,:,:] = dummy/torch.max(dummy)
        img_grid = torchvision.utils.make_grid(images,nrow=n)
        #print(images)
        
        self.writer.add_image("hidden weights",img_grid,global_step=step)
                
        

        