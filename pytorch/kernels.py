import torch

### RBF

def RBF_func(X1, X2, l):
  scaled_dist = (X1[:,None,:] - X2[None,:,:])/l.reshape(1,1,-1)
  
  return torch.exp(-0.5 * torch.square(scaled_dist))
  
