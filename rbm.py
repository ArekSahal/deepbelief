import numpy as np 
import torch 
from tqdm.auto import trange
from torch.utils.tensorboard import SummaryWriter

if torch.cuda.is_available():
    device = "cuda:0"
    #device = "cpu"
    print("Using GPU")
else:
    device = "cpu"
    print("Using CPU")

class BMNet:

    def __init__(self,n_visible, n_hidden,
    batch_size=10,
    learning_rate=0.01,
    momentum=0.5, 
    initial_momentum=0.5, # Initial momentum 
    weight_decay=0,
    sparsity_target=0.1,
    sparsity_penalty=0.05,
    cdn=1, # Amount of gibbsamples in our contrastive divergance
    tune_momentum=0.5, # Tune variables are used for fine tuning
    tune_decay=0.007, 
    top=False, # If the rbm is the top rbm in our DBN
    n_labels=0, # DBN
    tb=True, # Record data in tensorboard
    tb_funcs=[], # Functions from parent to run each epoch need to take parameter "step"
    writer=False, # Tensorboard writer
    part_of_dbn=False
    ):

        # Set number of nodes in the two layers
        self.n_visible = n_visible
        self.n_hidden = n_hidden

        # Initiate the weights and biases
        self.w = torch.normal(0,0.01,size=(n_visible, n_hidden)).float().to(device)
        self.w_r = None # Except these ones
        self.w_g = None # Fuck these ones
        self.visible_bias = torch.zeros(n_visible).to(device)
        self.hidden_bias = torch.zeros(n_hidden).to(device)
        self.hidden_bias_r = torch.zeros(n_hidden).to(device)
        self.visible_bias_g = torch.zeros(n_visible).to(device)

        # Acumulation variables - Store deltas for momentum
        self.delta_w_ac = torch.zeros(size=(n_visible,n_hidden)).to(device)
        self.delta_v_bias_ac = torch.zeros(n_visible).to(device)
        self.delta_h_bias_ac = torch.zeros(n_hidden).to(device)
        self.delta_w_g_ac = torch.zeros(size=(n_visible,n_hidden)).to(device)
        self.delta_w_r_ac = torch.zeros(size=(n_visible,n_hidden)).to(device)
        self.delta_v_bias_g_ac = torch.zeros(n_visible).to(device)
        self.delta_h_bias_r_ac = torch.zeros(n_hidden).to(device)
        self.q_ac = torch.zeros(self.n_hidden).to(device)

        # Set hyperparameters
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.initial_momentum = initial_momentum
        self.weight_decay = weight_decay
        self.tune_decay = tune_decay
        self.tune_momentum = tune_momentum
        self.cdn = cdn
        self.sparsity_target = sparsity_target
        self.sparsity_penalty = sparsity_penalty

        self.current_momentum = initial_momentum
        self.part_of_dbn = part_of_dbn

        # Save parameters to tensorboard
        if tb:
            self.params = {'lr': self.learning_rate,
            "bs": self.batch_size, 
            'wd': self.weight_decay,
            "m": self.momentum,
            "im": self.initial_momentum,
            "s": self.sparsity_target,
            "sp":self.sparsity_penalty
            }
            self.writer = writer
            self.tb_funcs = tb_funcs


        self.training_data = None
        self.validation_data = None
        self.batch_indx = [] # batch indexing [[batch_0_start, batch_0_end],...]

        self.train_subset = [] # A subset of the training data to comapre energies to validation data
        self.training_energies = []
        self.validation_energies = []

        self.top=top
        self.n_labels = n_labels

        
    def activation(self,x,T=1.0):
        """ Activation function:
            Sigmoid
            Ignore the T... for now..
        """
        return 1.0 / (1 + torch.exp(-x/T))

    def sample(self,X):
        """
        Given a matrix, samples the elements by comparing them to random floats.
        If the element is larger than a random float then it becomes 1 and 0 otherwise
        """
        sampled = torch.zeros(size=X.shape).to(device)
        rnd_mat = torch.from_numpy(np.random.rand(X.shape[0], X.shape[1])).to(device)
        sampled[rnd_mat <= X] = 1
        return sampled

    def v_to_h(self, V, sample=True, directed=False):
        """
        Forward pass
        V is assumed to be of shape (batch_size x n_visible)
        return H (batch_size x n_hidden)
        """
        if directed:
            output = torch.mm(V,self.w_r) + self.hidden_bias_r
        else:
            output = torch.mm(V,self.w) + self.hidden_bias
        H = self.activation(output)
        if sample:
            return self.sample(H)
        return H

    def h_to_v(self,H,sample=False, directed=False):
        """
        Backward pass
        H is assumed to be of shape (batch_size x n_hidden)
        return V (batch_size x n_visible)
        """
        if directed:
            output = torch.mm(H,self.w_g.t()) + self.visible_bias_g
        else:
            output = torch.mm(H,self.w.t()) + self.visible_bias
        if not self.top:
            V = self.activation(output)
        else:
            V = output
            lbs = torch.zeros(V.shape[0], self.n_labels)
            lbs_probs = torch.softmax(V[:,-self.n_labels:],dim=1)
            lbs[torch.arange(V.shape[0]),torch.argmax(lbs_probs,dim=1)] = 1
            base = self.activation(V[:,:-self.n_labels])
            if sample:
                base = self.sample(base)
            V[:,-self.n_labels:] = lbs
            V[:,:-self.n_labels] = base
        if sample and not self.top:
            return self.sample(V)
        return V


    def gibb_sample(self,X,n=1,sample=False):
        """
        Pretty self explanitory 
        """
        V = X
        for i in range(n-1):
            H = self.v_to_h(V,sample=True)
            V = self.h_to_v(H,sample=False)
        H = self.v_to_h(V,sample=True)
        V = self.h_to_v(H,sample=False)
        if sample:
            return self.sample(V)

        return V

    def set_data(self, X, val_X=False):
        """
        Data coming in is assumed to be shuffled.
        X = (n_samples x n_features)
        """
        self.training_data = X.to(device)
        if torch.is_tensor(val_X):
            self.validation_data = val_X.to(device)
            # Since the data is assumed to be shuffled we just take the first couple of data points
            self.train_subset = self.training_data[:self.validation_data.shape[0],:]
        else:
            self.train_subset = self.training_data[:int(0.05*self.training_data.shape[0]),:]

        # Create the batch indexing
        current_batch_iter = 0
        while current_batch_iter*self.batch_size<self.training_data.shape[0]:
            if (current_batch_iter + 1)*self.batch_size >= self.training_data.shape[0]:
                self.batch_indx.append([current_batch_iter*self.batch_size,-1]) # Make sure we get all of them
            else:
                self.batch_indx.append([current_batch_iter*self.batch_size , (current_batch_iter + 1)*self.batch_size])
            current_batch_iter += 1

    def reconstruct(self, X,n=1):
        """
        Takes a tensor X (n_samples x n_features) and return the output of gibbsampling
        """
        Y = self.gibb_sample(X,n)
        return Y

    def recon_loss(self, X):
        """
        Takes the mean norm as a measure of recognition loss
        """
        Y = self.reconstruct(X)
        diff = torch.norm(X - Y,dim=1)
        return torch.mean(diff)

    def train(self, epochs=1,save=False,filename="model",full_loss=False,test=False):
        """
        """

        # Start the tensorboard metrics
        self.write_to_tb(0,epochs)
        for ep in trange(epochs,desc="Training Layer"):
        #for ep in range(epochs):
            if ep >= 5:
                # Increase momentum after 5 epochs because Hinton et al. said so
                self.current_momentum = self.momentum

            # Shuffle training data
            rnd_index = np.arange(self.training_data.shape[0])
            np.random.shuffle(rnd_index)
            rnd_index = torch.from_numpy(rnd_index)
            data = self.training_data[rnd_index,:]

            for batch in trange(len(self.batch_indx), desc="Mini-batch",leave=False):
            #for batch in range(len(self.batch_indx)):
                current_batch = data[self.batch_indx[batch][0]:self.batch_indx[batch][1],:]

                # Forward pass
                v_0 = current_batch
                h_0 = self.v_to_h(v_0, sample=True)

                # Gibb sampling
                v_1 = self.gibb_sample(v_0, n=self.cdn,sample=False)

                # Backwards pass
                h_1 = self.v_to_h(v_1,sample=False)

                # Calculate deltas 
                delta_w, delta_v_bias, delta_h_bias = self.calc_deltas(v_0,h_0, v_1, h_1)

                # Calculate mean activation of hidden nodes
                q = torch.mean(h_0,dim=0)
                self.q_ac = self.q_ac*0.9 +(1 - 0.9)*q # Accumulating variable

                # Update weights
                self.update_weights(delta_w, delta_v_bias, delta_h_bias,lr=self.learning_rate)
        
            if save and ep %10 ==0:
                # Save just in case
                self.save_weights(filename)

            # Tensorboard stuff
            self.write_to_tb(ep+1,epochs)

        # Prepare for directed DBN
        self.untwine_weights()
        if not self.part_of_dbn:
            self.writer.add_hparams(self.params,{"hparam/Val_recon": self.recon_loss(self.validation_data)})
        return 0

    def save_weights(self, filename="model"):
        with open(filename + '_w.pt', 'wb') as f:
            torch.save(self.w,f)
        with open(filename + '_w_g.pt', 'wb') as f:
            torch.save(self.w_g,f)
        with open(filename + '_w_r.pt', 'wb') as f:
            torch.save(self.w_r,f)
        with open(filename + '_v_bias.pt', 'wb') as f:
            torch.save(self.visible_bias,f)
        with open(filename + '_v_bias_g.pt', 'wb') as f:
            torch.save(self.visible_bias_g,f)
        with open(filename + '_h_bias_r.pt', 'wb') as f:
            torch.save(self.hidden_bias_r,f)
        with open(filename + '_h_bias.pt', 'wb') as f:
            torch.save( self.hidden_bias,f)


    def load_weights(self,filename="model"):
        with open(filename + '_w.pt', 'rb') as f:
            self.w = torch.load(f,map_location=torch.device(device))
        try:
            with open(filename + '_w_r.pt', 'rb') as f:
                self.w_r = torch.load(f,map_location=torch.device(device)).float()
            with open(filename + '_w_g.pt', 'rb') as f:
                self.w_g = torch.load(f,map_location=torch.device(device)).float()
        except:
            self.w_g = self.w.clone()
            self.w_r = self.w.clone()
        with open(filename + '_v_bias.pt', 'rb') as f:
            self.visible_bias = torch.load(f,map_location=torch.device(device)).float()
        with open(filename + '_v_bias_g.pt', 'rb') as f:
            self.visible_bias_g = torch.load(f,map_location=torch.device(device)).float()
        with open(filename + '_h_bias_r.pt', 'rb') as f:
            self.hidden_bias_r = torch.load(f,map_location=torch.device(device)).float()
        with open(filename + '_h_bias.pt', 'rb') as f:
            self.hidden_bias = torch.load(f,map_location=torch.device(device)).float()


    def untwine_weights(self):
        """
        Prepare for DBN
        """
        self.w_g = self.w.clone()
        self.w_r = self.w.clone()
        self.visible_bias_g = self.visible_bias.clone()
        self.hidden_bias_r = self.hidden_bias.clone()

        
    def update_weights(self,delta_w, delta_v_bias, delta_h_bias,lr):

        sparsity = (self.q_ac - self.sparsity_target)*self.sparsity_penalty 
        self.delta_v_bias_ac = self.delta_v_bias_ac*self.momentum + delta_v_bias*lr 
        self.delta_h_bias_ac = self.delta_h_bias_ac*self.momentum + (delta_h_bias - sparsity)*lr 
        self.delta_w_ac = self.delta_w_ac*self.momentum + lr*(delta_w - self.weight_decay*self.w - sparsity) 

        self.w += self.delta_w_ac 
        self.visible_bias += self.delta_v_bias_ac
        self.hidden_bias += self.delta_h_bias_ac 

    def update_w_g(self, X,Y,Y_tilde, tune_rate):
        self.delta_w_g_ac = self.delta_w_g_ac*self.tune_momentum + tune_rate*(torch.mm((Y - Y_tilde).t(),X)/float(X.shape[0]) - self.w_g*self.tune_decay) # I know this looks bad but its legit. Or at least there is no error.
        self.delta_v_bias_g_ac = self.delta_v_bias_g_ac*self.tune_momentum + tune_rate*torch.mean(Y - Y_tilde,dim=0)

        self.w_g = self.w_g + self.delta_w_g_ac
        self.visible_bias_g = self.visible_bias_g + self.delta_v_bias_g_ac

    def update_w_r(self,X,Y,Y_tilde,tune_rate):
        self.delta_w_r_ac = self.delta_w_r_ac*self.tune_momentum + tune_rate*(torch.mm(X.t(),(Y- Y_tilde))/float(X.shape[0]) - self.w_r*self.tune_decay)
        self.delta_h_bias_r_ac = self.delta_h_bias_r_ac*self.tune_momentum + tune_rate*torch.mean(Y - Y_tilde,dim=0)
        self.w_r = self.w_r + self.delta_w_r_ac
        self.hidden_bias_r = self.hidden_bias_r + self.delta_h_bias_r_ac

    def calc_deltas(self,v_0,h_0,v_1,h_1):
        M_1 = torch.mm(v_1.t(),h_1)/float(v_1.shape[0])
        M_0 = torch.mm(v_0.t(),h_0)/float(v_0.shape[0])

        delta_w = M_0 - M_1
        delta_v_bias = torch.mean(v_0 - v_1,dim=0)
        delta_h_bias = torch.mean(h_0 - h_1,dim=0) 

        return delta_w, delta_v_bias, delta_h_bias

    def energy(self, V, tuned=False):
        """
        Calculate F(V) in exp(-F(V))/Z which gives the probability of vector V.
        The output is not the real probability since we dont know the partition function
        but we can use it to compare between data.
        Lower value means better. Don't @ me.
        """
        if tuned:
            v_b = self.visible_bias_g
            h_b = self.hidden_bias_r
            w = self.w_r
        else:
            v_b = self.visible_bias
            h_b = self.hidden_bias
            w = self.w
        
        X = torch.mm(V,w) + h_b
        #print(torch.mv(V,v_b).shape, torch.log(1 + torch.exp(X)).shape)
        F = - torch.mv(V,v_b) - torch.sum(torch.log(1 + torch.exp(X)), dim=1)
        return F

    def write_to_tb(self,ep,eps):
        self.writer.add_histogram("Weights", self.w.flatten(), global_step=ep)
        self.writer.add_histogram("Delta weights", torch.abs(self.delta_w_ac.flatten()), global_step=ep)
        self.writer.add_histogram("Visible bias", self.visible_bias.flatten(), global_step=ep)
        self.writer.add_histogram("Hidden bias", self.hidden_bias.flatten(), global_step=ep)
        if torch.is_tensor(self.validation_data):
            self.writer.add_scalars("Free energy", {"Training": torch.mean(self.energy(self.train_subset)), "Validation": torch.mean(self.energy(self.validation_data))}, global_step=ep)
            self.writer.add_scalars("Reconstruction Loss", {"Training": self.recon_loss(self.train_subset), "Validation": self.recon_loss(self.validation_data)}, global_step=ep)
        else:
            self.writer.add_scalars("Free energy", {"Training": torch.mean(self.energy(self.train_subset))}, global_step=ep)
            self.writer.add_scalars("Reconstruction Loss", {"Training": self.recon_loss(self.train_subset)}, global_step=ep)
        for f in self.tb_funcs:
            f(step=ep,eps=eps)


                
