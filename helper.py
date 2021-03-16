import numpy as np
import torch
def sample(X):
    #print("X = ", X)
    sampled = torch.zeros(size=X.shape)
    rnd_mat = torch.from_numpy(np.random.rand(X.shape[0], X.shape[1]))
    #print(sampled.shape, rnd_mat.shape, X.shape)
    sampled[rnd_mat <= X] = 1
    #print(np.sum(sampled == 0))
    #print("S = ", sampled)
    return sampled

def softmax(support):

    """ 
    Softmax activation function that finds probabilities of each category
        
    Args:
      support: shape is (size of mini-batch, number of categories)      
    Returns:
      probabilities: shape is (size of mini-batch, number of categories)      
    """

    expsup = torch.exp(support - torch.max(support,dim=1))
    return expsup / torch.sum(expsup,dim=1)
