import torch
import numpy
print(torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x = torch.Tensor([1,2,3]).to(device)
y = torch.Tensor([4,5,6]).to(device)
x = x+y
print(x)