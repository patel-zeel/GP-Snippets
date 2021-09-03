from kernels import RBF_func

class GPRegressor(torch.nn.Module):
  def __init__(self, X, y):
    super(GPRegressor, self).__init__()
    self.X = X
    self.y = y
    self.pi = torch.tensor(np.pi)
    
    self.length_scale = torch.nn.Parameter(torch.ones(1, requires_grad=True))
    self.std = torch.nn.Parameter(torch.ones(1, requires_grad=True))
    self.noise_std = torch.nn.Parameter(torch.ones(1, requires_grad=True))
  
  def rbf_kernel(self, Xi, Xj):
    X1 = X1/self.length_scale.reshape(1,-1)
    X2 = X2/self.length_scale.reshape(1,-1)
    scaled_dist = (X1[:,None,:] - X2[None,:,:]).sum(dim=2)
  
    return torch.exp(-0.5 * torch.square(scaled_dist))

  def nlml(self):
    self.K_nn = self.rbf_kernel(self.X, self.X)
    diag = self.K_nn.diagonal()
    diag += self.noise_std**2

    self.L = torch.linalg.cholesky(self.K_nn)
    self.alpha_ = torch.cholesky_solve(y, self.L)
    
    return 0.5 * (self.y.T@self.alpha_ + torch.sum(torch.log(self.L.diagonal())) + self.y.shape[0] * 2 * self.pi)/self.X.nelement()

  def predict(self, X):
    K_tn = self.rbf_kernel(X, self.X)
    y_mean = K_tn@self.alpha_

    v = torch.cholesky_solve(K_tn.T, self.L)
    y_cov = self.rbf_kernel(X, X) - K_tn@v
    diag = y_cov.diagonal()
    diag += self.noise_std**2

    return y_mean, y_cov
