import os
import torch.distributed.rpc as rpc
import torch

import torchvision.transforms as transforms
from PIL import Image
from torch.nn.modules.container import Sequential
import torch.nn as nn
import torchvision.models as models

os.environ['MASTER_ADDR'] = '134.193.129.157'
os.environ['MASTER_PORT'] = '8880'

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return x
        


r_50=models.resnet50()
r_models=[r_50]
for r in r_models:
    net1=[]
    for module in r.children():
      if isinstance(module,Sequential):
       for m in module.children():
        net1.append(m)
  
      else:
       net1.append(module)

    len_net1=len(net1)
    
    
split_point=8
net1=nn.Sequential(*net1)
left=list(net1.children())[:split_point]
resnet_left=nn.Sequential(*left)
resnet_right= list(net1.children())[split_point:-1]
resnet_right = nn.Sequential(*[*resnet_right, Flatten(), list(net1.children())[-1]])

device='cuda:2'
       
        
def my_script_add(t1, t2):
    return t1+t2
 

rpc.init_rpc("worker1", rank=1, world_size=2)
print("success")

out_left = rpc.rpc_sync("worker0", my_script_add, args=(torch.ones(2), 3))

print(type(out_left))
print(out_left)

resnet_right.to(device)
out_left=out_left.to(device)
out_right=resnet_right(out_left)
#out_right=out_right.cpu()
print(out_right)
 
rpc.shutdown()


