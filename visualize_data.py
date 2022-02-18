from dataclasses import dataclass
from dataset import SHREC2022Primitives, minkowski_collate
import transforms as t
import torch
from torch.utils.data import DataLoader
from networks import MinkowskiFCNN
import MinkowskiEngine as ME
from tqdm import tqdm
import os
path = "/home/ioannis/Desktop/programming/data/SHREC/SHREC2022/dataset"


import open3d as o3d

def makeO3Dpc(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.numpy())
    return pcd

def visualize_pointcloud(pcd):
    if not isinstance(pcd, o3d.geometry.PointCloud):
        pcd = makeO3Dpc(pcd)
    o3d.visualization.draw_geometries([pcd])

train_transforms = [
                    #t.Translate(), 
                    #t.SphereNormalization(), 
                    #t.RandomRotate(180, 0),
                    #t.RandomRotate(180, 1),
                    #t.RandomRotate(180, 2),
                    #t.GaussianNoise()
                    ]

t_dataset = SHREC2022Primitives(path, train=True, valid=False, valid_split=0.2, transform=train_transforms)

idxes = torch.floor(torch.rand(10, 1) * len(t_dataset)).long()


for idx in idxes:
    print(t_dataset[idx].keys())
    visualize_pointcloud(t_dataset[idx]["x"])
