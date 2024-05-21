import torch 

MU = 5000

def u_law(x):
    eps = torch.ones_like(x)
    return torch.log(eps + MU * x) / torch.log(eps + MU)

def u_law_inverse(x):
    eps = torch.ones_like(x)
    return ((eps + MU) ** x - 1) / MU
