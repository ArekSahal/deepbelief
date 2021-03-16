import numpy as np
import torch
from bm_torch import BMNet
from helper import sample
from tqdm.auto import trange
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

if torch.cuda.is_available():
    device = "cuda:0"
    #device = "cpu"
    #print("Using GPU")
else:
    device = "cpu"
    #print("Using CPU")

class DBNet:

    def __init__(self,n_visible, hidden, 
    n_labels,batch_size=20,
    learning_rate=0.05,
    momentum=0.7,
    initial_momentum=0.5,
    weight_decay=0.0001,
    tune_rate=0.05, 
    tune_momentum=0.5, 
    tune_decay=0.007, 
    cdns=False,
    sparsity_penalty=0.0,
    sparsity_target=0.1,
    path_writer="runs/DBN"):

        self.n_visible = n_visible
        self.n_hidden = len(hidden)
        self.n_labels = n_labels
        self.hidden = hidden

        self.tune_rate = tune_rate
        self.tune_momentum=tune_momentum
        self.tune_decay=tune_decay
        self.batch_size = batch_size

        self.writer = SummaryWriter(path_writer)
        self.params = {'lr': learning_rate,
                        "bs": batch_size, 
                        'wd': weight_decay,
                        "m": momentum,
                        "im": initial_momentum,
                        "s": sparsity_target,
                        "sp": sparsity_penalty,
                        "tr": tune_rate,
                        "tm": tune_momentum,
                        "td": tune_decay,
            
            }
    
        cdn = 1
        if cdns:
            cdn = cdns[0]
        self.stack = [BMNet(n_visible,hidden[0],batch_size=self.batch_size,learning_rate=learning_rate,momentum=momentum,initial_momentum=initial_momentum,weight_decay=weight_decay, tune_momentum=tune_momentum, tune_decay=tune_decay, cdn=cdn,sparsity_penalty=sparsity_penalty,sparsity_target=sparsity_target,part_of_dbn=True)]
        for m in range(0,self.n_hidden-1):
            if m >= self.n_hidden - 2:
                cdn = 1
                if cdns:
                    cdn = cdns[m+1]
                self.stack.append(BMNet(hidden[m] + n_labels, hidden[m+1], batch_size=self.batch_size,learning_rate=learning_rate,momentum=momentum,initial_momentum=initial_momentum,weight_decay=weight_decay, tune_momentum=tune_momentum, tune_decay=tune_decay, cdn=cdn,top=True,n_labels=self.n_labels,sparsity_penalty=sparsity_penalty,sparsity_target=sparsity_target,part_of_dbn=True))
            else:
                cdn = 1
                if cdns:
                    cdn = cdns[m+1]
                self.stack.append(BMNet(hidden[m], hidden[m+1], batch_size=self.batch_size,learning_rate=learning_rate,momentum=momentum,initial_momentum=initial_momentum,weight_decay=weight_decay, tune_momentum=tune_momentum, tune_decay=tune_decay, cdn=cdn,sparsity_penalty=sparsity_penalty,sparsity_target=sparsity_target,part_of_dbn=True))

        for m in range(len(self.stack)):
            self.stack[m].writer = SummaryWriter(path_writer+"/layer_" +str(m))
        self.training_data = []
        self.validation_data = []
        self.labels = []
        self.validation_labels = []
        self.val_energies = []
        self.train_energies = []
        self.val_recons = []
        self.train_recons = []

        self.batch_indx = []


    def set_training_data(self,X,labels,split=0.8):
        if split>=1.0:
            self.training_data = X.to(device)
            self.labels = labels.to(device)
            
        else:
            shuffl_index = np.arange(X.shape[0])
            np.random.shuffle(shuffl_index)
            data = X[shuffl_index,:]
            lbs = labels[shuffl_index,:]

            lmt = int(X.shape[0]*split)

            self.training_data = data[:lmt,:].to(device)
            self.labels = lbs[:lmt,:].to(device)

            self.validation_data = data[lmt:,:].to(device)
            self.validation_labels = lbs[lmt:,:].to(device)
            self.validation_labels_persistent = self.validation_labels.clone()
            self.validation_data_persistent = self.validation_data.clone()
            self.labels_persistent = self.labels.clone()
            self.training_data_persistent = self.training_data.clone()

        current_batch_iter = 0
        while current_batch_iter*self.batch_size<self.training_data.shape[0]:
            if (current_batch_iter + 1)*self.batch_size >= self.training_data.shape[0]:
                self.batch_indx.append([current_batch_iter*self.batch_size,-1])
                
            else:
                self.batch_indx.append([current_batch_iter*self.batch_size , (current_batch_iter + 1)*self.batch_size])
            current_batch_iter += 1

    def train(self, epochs,save=True, filename="models/db_", test=False, full_loss=True):
        data = self.training_data.clone()
        val_data = self.validation_data.clone()
        for m in trange(len(self.stack),desc="Greedy layer-by-layer training"):
        #for m in range(len(self.stack)):

            if m >=len(self.stack)-1:
                #print("Training last layer")
                lbs = self.labels.clone()
                data = torch.hstack([data, lbs]).float()
                val_data = torch.hstack([val_data, self.validation_labels]).float()
                self.stack[-1].set_data(data ,val_data)
                self.stack[-1].train(epochs,save=save,filename=filename + str(self.n_hidden-1),test=test, full_loss=full_loss)
                if save:
                    self.stack[-1].save_weights(filename=filename + str(self.n_hidden-1))
            else:
                self.stack[m].set_data(data, val_data)
                self.stack[m].train(epochs,save=save, filename=filename + str(m),test=test,full_loss=full_loss)
                if save:
                    self.stack[m].save_weights(filename=filename + str(m))
                # ---------- CHANGE HERE ------------
                data = self.stack[m].v_to_h(data,sample=False)
                val_data = self.stack[m].v_to_h(val_data,sample=False)
                # -----------------------------------
        self.params["epochs"] = epochs
        self.writer.add_hparams(self.params,{"hparam/Accuracy":self.validate()})

    def load_weights(self,filename="models/db_"):
        for i in range(len(self.stack)):
            self.stack[i].load_weights(filename+str(i))

    def predict(self,X):
        data = X
        data = data.to(device)
        for m in range(len(self.stack) - 1):
            # ----------------- CHANGE HERE -----------------
            data = self.stack[m].v_to_h(data, directed=True, sample=False)
            # ---------------------------------------------
        
        lb = torch.ones((X.shape[0],self.n_labels)).to(device) /self.n_labels
        data_pers = data.clone()
        data = torch.hstack([data, lb])
        #print(data[0,-self.n_labels:])
        #------- MIN IDE --------
        """
        for i in range(20):
            h = self.stack[-1].v_to_h(data, sample=True)
            data = self.stack[-1].h_to_v(h)
            data[:,:-self.n_labels] = data_pers
        out = data
        
        #------- slut på min ide ---------
        """
        out = self.stack[-1].gibb_sample(data, n=20)
        #print(out[0,-self.n_labels:])
        #quit()
        #out = out.to("cpu")
        return out[:,-self.n_labels:]

    def generate(self, label,n=200):
        n_iters = n
        iters = []
        lb = label.to(device)
        data = torch.rand(self.stack[-1].n_visible - self.n_labels)
        data = sample(torch.stack([data]))
        data = torch.hstack([data[0,:], label])
        data = torch.stack([data]).to(device)
        for i in range(n_iters):
            data = self.stack[-1].gibb_sample(data,n=1,sample=True)
            #data = sample(data)
            inter_data = data[:,:-self.n_labels]
            for i in torch.arange(len(self.stack)-2, -1,-1):
                if i > 0:
                    inter_data = self.stack[i].h_to_v(inter_data, sample=True, directed=True)
                else:
                    inter_data = self.stack[i].h_to_v(inter_data, sample=False, directed=True)
            iters.append(inter_data)
            data[0,-self.n_labels:] = lb

        return iters

    def validate(self,method="validation"):
        if method=="validation":
            data = self.validation_data.to(device)
            preds = self.predict(data)
            count = torch.sum(torch.argmax(preds,dim=1) == torch.argmax(self.validation_labels,dim=1)).tolist() / preds.shape[0]
            return count
        elif method=="training":
            data = self.training_data[:10000,:]
            data = data
            preds = self.predict(data)
            count = 0
            for i in range(self.labels[:10000,:].shape[0]):
                if torch.argmax(preds[i,:]) == torch.argmax(self.labels[i,:]):
                    count += 1
            return count/self.labels[:10000,:].shape[0]


    def wakesleep_finetune(self, epochs,save=True,filename="models/db_"):
        #print("Fine-tuning")
        if torch.is_tensor(self.validation_data):
            self.writer.add_scalar("Tuning accuracy: ",self.validate(),global_step=0)
        for ep in trange(epochs,desc="Sleep-Wake fine-tuning"):
        #for ep in range(epochs):
            # Set up minibatch
            rnd_index = np.arange(self.training_data.shape[0])
            np.random.shuffle(rnd_index)
            data = self.training_data[rnd_index,:]
            lbs = self.labels[rnd_index,:]
            
            data = data
            lbs = lbs

            for batch in trange(len(self.batch_indx), desc="Mini-batch",leave=False):
            #for batch in range(len(self.batch_indx)):
                current_batch = data[self.batch_indx[batch][0]:self.batch_indx[batch][1],:]
                X = current_batch
                current_lbs = lbs[self.batch_indx[batch][0]:self.batch_indx[batch][1],:]
                Vs_wake = [X]
                
                #Wake pass
                for m in range(len(self.stack)-1):
                    Vs_wake.append(self.stack[m].v_to_h(Vs_wake[-1], sample=True, directed=True))
                
                #Top sampling
                X_and_lb = torch.hstack([Vs_wake[-1], current_lbs])
                h_0 = self.stack[-1].v_to_h(X_and_lb,sample=True)
                Top_v = self.stack[-1].gibb_sample(X_and_lb,n=3)
                Top_h = self.stack[-1].v_to_h(Top_v, sample=True)

                #Sleep pass
                Vs_sleep = [Top_v[:,:-self.n_labels]]
                for m in np.arange(len(self.stack)-2,-1,-1):
                    if m>0:
                        Vs_sleep.append(self.stack[m].h_to_v(Vs_sleep[-1], sample=True, directed=True))
                    else:
                        Vs_sleep.append(self.stack[m].h_to_v(Vs_sleep[-1], sample=False, directed=True))

                #Calculations
                delta_w, delta_v_bias , delta_h_bias = self.stack[-1].calc_deltas(X_and_lb, h_0,Top_v,Top_h)

                #Update params
                for m in range(len(self.stack)-1):
                    phidprops = self.stack[m].h_to_v(Vs_wake[m+1],sample=False, directed=True)
                    self.stack[m].update_w_g(Vs_wake[m+1], Vs_wake[m], phidprops, self.tune_rate)

                    psleeppenstate = self.stack[m].v_to_h(Vs_sleep[len(Vs_sleep) -1-m],sample=False,directed=True)
                    self.stack[m].update_w_r(Vs_sleep[len(Vs_sleep) -1-m],Vs_sleep[len(Vs_sleep) -2-m],psleeppenstate,self.tune_rate)

                self.stack[-1].update_weights(delta_w, delta_v_bias , delta_h_bias,lr=self.tune_rate)
            if save:
                for m in range(len(self.stack)):
                    self.stack[m].save_weights(filename=filename + str(m))
            if torch.is_tensor(self.validation_data):
                self.writer.add_scalar("Tuning accuracy: ",self.validate(),global_step=ep+1)
        self.writer.add_hparams(self.params,{"hparam/Tune_Accuracy":self.validate()})
        if save:
            for m in range(len(self.stack)):
                self.stack[m].save_weights(filename=filename + str(m))
