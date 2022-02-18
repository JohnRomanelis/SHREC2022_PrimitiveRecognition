import os
import torch
from tqdm import tqdm

sz = len(os.listdir("/home/ioannis/Desktop/programming/data/SHREC/SHREC2022/dataset/training/GTpointCloud/"))

print(sz)
path = lambda i: f"/home/ioannis/Desktop/programming/data/SHREC/SHREC2022/dataset/training/GTpointCloud/GTpointCloud{i}.txt"


for i in tqdm(range(1, sz)):

    with open(path(i), "r") as F:

        contents = F.readlines()

        axis = None

        if int(contents[0][0]) == 1:

            ax = list(map(float, contents[1:4]))
            axis = torch.Tensor(ax)

        elif int(contents[0][0]) == 2:
        
            ax = list(map(float, contents[2:5]))
            axis = torch.Tensor(ax)

        elif int(contents[0][0]) == 4:

            ax = list(map(float, contents[2:5]))
            axis = torch.Tensor(ax)

        elif int(contents[0][0]) == 5:
            
            ax = list(map(float, contents[3:6]))
            axis = torch.Tensor(ax)

        elif int(contents[0][0]) == 3:
            axis = torch.Tensor([1,0,0])
        
        length = (axis * axis).sum(-1).sqrt()

        if length.item() < 0.99999:

            print("i has length: ", length)

