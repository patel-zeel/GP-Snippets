import torch

### RBF

def RBF_func(X1, X2, l):
    X1 = X1/l.reshape(1,-1)
    X2 = X2/l.reshape(1,-1)
    scaled_dist = (X1[:,None,:] - X2[None,:,:])
  
    return torch.exp(-0.5 * torch.square(scaled_dist))
  
