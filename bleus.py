import torch

current_device = torch.cuda.current_device()
print('current_device = ', current_device)

count = torch.cuda.device_count()
print('count = ', count)

for i in range(count):
    name = torch.cuda.get_device_name(0)
    print('name = ', name)
