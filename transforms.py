import torch


class GetMean():
    
    def __call__(self, x):
        x["mean"] = x['x'].mean(0)

        return x

class Initialization():

    def __call__(self, x):

        x["inverse_rotation"] = torch.eye(3)
        return x

class SphereNormalization():
    
    def __init__(self, scale=1):
        self.scale = scale
        
    def __call__(self, x):
        
        assert isinstance(x["x"], torch.Tensor)
        assert x["x"].dim() == 2
        
        max_norm = (x["x"]*x["x"]).sum(-1).max().sqrt() / self.scale
        x["x"] /= max_norm
        
        x["norm_factor"] = max_norm
        
        return x

class Translate():
    
    def __init__(self, center=[0,0,0]):
        self.center = torch.Tensor(center)
        assert self.center.dim() == 1
        assert self.center.shape[0] == 3
    
    def __call__(self, x):
        #print(x["x"])
        assert isinstance(x["x"], torch.Tensor)
        assert x["x"].dim() == 2
        
        centroid = x["x"].mean(dim=0)
        shift = - centroid + self.center
        x["x"] = x["x"] + shift
        
        x["shift"] = shift
        
        return x

class RandomRotate():
    
    def __init__(self, max_deg=20, axis=0):
        '''
            axis: 0->x, 1->y, 2->z
        '''
      
        self.axis = axis
        self.max_deg = max_deg
        assert axis in [0,1,2]
        
    def __call__(self, x):
        assert isinstance(x["x"], torch.Tensor)
        assert x["x"].dim() == 2
        
        rad = torch.rand(1).item() * self.max_deg * 0.0174532925
        trans = self._R(rad)
        
        x["rotation"] = trans
        x["inverse_rotation"] = x["inverse_rotation"] @ self._R(-rad)
        x["x"] = (trans.unsqueeze(0) @ x["x"].unsqueeze(-1)).squeeze(-1)
        
        return x

    
    def _R(self, rad):
        
        import math
        
        if self.axis == 0:
            
            return torch.Tensor([
                [1,             0,              0],
                [0, math.cos(rad), -math.sin(rad)],
                [0, math.sin(rad),  math.cos(rad)]
            ])
        
        elif self.axis == 1:
            
            return torch.Tensor([
                [math.cos(rad),  0, math.sin(rad)],
                [0,              1,             0],
                [-math.sin(rad), 0, math.cos(rad)]
            ])
        
        else:
            
            return torch.Tensor([
                [math.cos(rad), -math.sin(rad), 0],
                [math.sin(rad), math.cos(rad),  0],
                [0,             0,              1]
            ])
    
class GaussianNoise():
    
    def __init__(self, mean=0, std=.01):
        
        self.mean = mean
        self.std = std
    
    def __call__(self, x):
        
        assert isinstance(x["x"], torch.Tensor)
        assert x["x"].dim() == 2
        
        N, d = x["x"].shape
        noise = torch.normal(mean=self.mean, std = self.std, size=(N, d))
        x["noise"] = {"mean": self.mean, "std": self.std}
        x["x"] = x["x"] + noise
        
        return x