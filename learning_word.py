#%%
import torch
from opt_einsum import contract

# %%
data = torch.arange(0,10)
print(data)
torch.index_select(data, dim=0, index=torch.tensor([1, 2]))
# %%
data = torch.arange(0,9).view(3,3)
print(data)
print(torch.index_select(data, dim=1, index=torch.tensor([1,2])))
# %%
x = torch.arange(0,6).view(1,2,3)
y = torch.arange(0,15).view(1,3,5)
print(x)
print(y)
print(contract('bld,brl->brd', y, x))
# %%
