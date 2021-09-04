import torch

### RBF

def RBF_func(Xi, Xj, l):
    Xi = Xi/l.reshape(1,-1)
    Xj = Xj/l.reshape(1,-1)
    scaled_dist = (Xi[:,None,:] - Xj[None,:,:])
  
    return torch.exp(-0.5 * torch.square(scaled_dist))
  
