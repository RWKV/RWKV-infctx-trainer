import torch

w = torch.tensor([0.9,0.5])
x = torch.tensor([[1,1],[1,1],[1,1.]]).t()

from cumsumwithdecay import discounted_cumsum_left, discounted_cumsum_right

print(discounted_cumsum_right(x,w).t())
print(discounted_cumsum_left(x,w).t())