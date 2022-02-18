from re import A
import MinkowskiEngine as ME
import torch 
import einops
import os

def norm(x):
    return (x * x).sum(-1).sqrt()

### ---- UTILS ---- ###
def parse_plane(lines):
    
    normal = torch.Tensor(list(map(float, lines[1:4])))
    assert norm(normal) > 0.9999

    vertex = torch.Tensor(list(map(float, lines[4:])))
    data = torch.Tensor([0] + list(map(float, lines[1:])) + [-1, -1])

    return {"type": "plane", "class": 0, "vertex": vertex, "normal":normal, "data": data}

def parse_cylinder(lines):
    
    radius = float(lines[1])
    axis = torch.Tensor(list(map(float, lines[2:5])))
    assert norm(axis) > 0.9999
    vertex = torch.Tensor(list(map(float, lines[5:])))
    data = torch.Tensor([1] + list(map(float, lines[1:])) + [-1])

    return {"type": "cylinder", "class": 1, "radius": radius, "axis": axis, "vertex": vertex, "data": data}

def parse_sphere(lines):
    
    radius = float(lines[1])
    center = torch.Tensor(list(map(float, lines[2:])))
    data = torch.Tensor([2] + list(map(float, lines[1:])) + [-1]*4)

    return {"type": "sphere", "class": 2, "radius": radius, "center": center, "data": data}

def parse_cone(lines):
    
    angle = float(lines[1])
    axis = torch.Tensor(list(map(float, lines[2:5])))
    assert norm(axis) > 0.9999
    vertex = torch.Tensor(list(map(float, lines[5:])))
    data = torch.Tensor([3] + list(map(float, lines[1:])) + [-1])

    return {"type": "cone", "class": 3, "angle": angle, "axis": axis, "vertex": vertex, "data": data}

def parse_torus(lines):
    
    major_radius = float(lines[1])
    minor_radius = float(lines[2])
    axis = torch.Tensor(list(map(float, lines[3:6])))
    assert norm(axis) > 0.9999
    center = torch.Tensor(list(map(float, lines[6:])))
    data = torch.Tensor([4] + list(map(float, lines[1:])))

    return {"type": "torus", "class": 4, "major_radius": major_radius, "minor_radius": minor_radius, "axis": axis, "center": center, "data": data}

def parse_point_cloud(fname):

    file = open(fname)
    points = []

    for line in file.readlines():
        pts = torch.Tensor(list(map(float, line.split(","))))
        points.append(pts)

    return einops.rearrange(points, "n d -> n d")

def parse_label(fname):
    
    file = open(fname)
    
    #assigning a distinct function to handle each type of primitive
    handlers ={
                "1": parse_plane,
                "2": parse_cylinder,
                "3": parse_sphere,
                "4": parse_cone,
                "5": parse_torus
                }
    
    #parsing the contents of the file. The first character corresponds to a specific type of primitive
    contents =  file.readlines()
    
    #handling the primitive and returning the label
    return handlers[contents[0][0]](contents)

### --------------------- ####
### ------ Dataset ------ ####
### --------------------- ####

class SHREC2022Primitives(torch.utils.data.Dataset):
    
    def __init__(self, path, train=True, valid=False, valid_split=0.2, transform = [], category="all"):
        
        assert category in ["all", "plane", "cylinder", "sphere", "cone", "torus"]
        self.shape_indices = {
            "all": [0,1,2,3,4,5],
            "plane": [0],
            "cylinder": [1],
            "cone": [2],
            "sphere": [3],
            "torus": [4]
        }

        self.path = os.path.join(path, "training" if train else "test")
        self.pc_prefix = "pointCloud"
        self.gt_prefix = "GTpointCloud"
        self.format = ".txt"
        self.valid = valid if train else False
        self.size = 0 if train else len(os.listdir(self.path))
        self.transform = transform
        
        #check if an existing train-validation split matches the one given
        split_info_file = os.path.join(path, "training/split_info.txt")
        self.t_savefile = os.path.join(path, "training/train_split.txt")
        self.v_savefile = os.path.join(path, "training/valid_split.txt")
        if os.path.exists(split_info_file):
            with open(split_info_file) as F:
                v, vsize, tsize, c = list(map(float, F.readline().split(',')))

                if v == valid_split and self.shape_indices[category][-1] == c:
                    print("Specified split already exists. Using the existing one.")
                    self.size = int(vsize if self.valid else tsize)
                    return
        
        if train:
            print("Creating a new train-validation split.")
            import random
            with open(os.path.join(path, "training/indices.txt")) as F, open(self.t_savefile, "w") as T,\
                open(self.v_savefile, "w") as V, open(split_info_file, "w") as I:

                lines = F.readlines()
                cat_sz = len(lines[0].split(","))
                train_sz = int(cat_sz * (1-valid_split))

                v_indices = []
                t_indices = []

                for i, line in enumerate(lines):
                    if i not in self.shape_indices[category]:
                        continue
                    line = list(map(int, line.split(",")))
                    random.shuffle(line)
                    v_indices = v_indices + line[train_sz:]
                    t_indices = t_indices + line[:train_sz]

                self.size = len(v_indices) if valid else len(t_indices)
                T.write("\n".join(map(str, t_indices)))
                V.write("\n".join(map(str, v_indices)))
                
                I.write(
                          str(valid_split)+
                    ',' + str(len(v_indices))+
                    ',' + str(len(t_indices)) +
                    ',' + str(self.shape_indices[category][-1])
                )
    
    def __getitem__(self, index):
        
        with open(self.v_savefile if self.valid else self.t_savefile, "r") as F:
            index = F.readlines()[index]
            index = int(index) if '\n' not in index else int(index[:-1])
         
        
        #assembling the file name for the data and labels
        pc_name = os.path.join(self.path, self.pc_prefix, self.pc_prefix + str(index) + self.format)
        gt_name = os.path.join(self.path, self.gt_prefix, self.gt_prefix + str(index) + self.format)
        
        #parsing the point cloud
        pcloud = parse_point_cloud(pc_name)
        label = parse_label(gt_name)

        data = {"x": pcloud, "y": label['data']}
        
        for t in self.transform:
            data = t(data)
        
        return data
        
    def __len__(self):
        
        return self.size
    

def minkowski_collate(list_data):
    coordinates, features, labels = ME.utils.sparse_collate(
        [d['x'] for d in list_data],
        [d['x'] for d in list_data],
        [d['y'][0].unsqueeze(0) for d in list_data],
        dtype = torch.float32
    )
    
    return {
        "coordinates": coordinates, 
        "features"   : features,
        "labels"     : labels
    }

if __name__ == "__main__":

    pass

    ##
    #
    # *********** THYMISOU NA SVHSEIS TO ARXEIO split_info.txt KAI NA XANAKANEIS INITIALIZE TO DATASET
    # EXW VALEI ENA EXTRA OPTION POU UPODHLWNEI TO SHAPE POU EPILEXAME, KAI STO TWRINO SPLIT DEN UAPRXEI STO ARXEIO