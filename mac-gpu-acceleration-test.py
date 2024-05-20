import torch
print('start')
x = torch.ones(5, device="cpu")
print(x)
print('start')
x = torch.ones(5, device="mps")
print(x)