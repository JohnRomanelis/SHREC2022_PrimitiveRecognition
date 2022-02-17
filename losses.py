import torch

class Losses():
        
    def AxisToAxisLoss(self, ax1, ax2):
        
        '''
            ax1: Bx3
            ax2: Bx3
        '''
        
        B = ax1.shape[0]
        
        #normalizing
        ax1 = ax1 / self.norm(ax1) 
        ax2 = ax2 / self.norm(ax2)
        
        return  (torch.ones(B) - (ax1.unsqueeze(1) @ ax2.unsqueeze(-1)).abs().squeeze(-1)).mean()
        
    def PointToPointLoss(self, p1, p2):
        
        '''
            p1: Bx3
            p2: Bx3
        '''
        q = p1 - p2
        return (q * q).sum(-1).mean()
    
    def PointToAxisLoss(self, p, ax, v):
        
        '''
            p:  Bx3 predicted point
            ax: Bx3 actual axis
            v:  Bx3 a point on the actual axis
        '''
        
        dist = self.norm(torch.cross(p-v, p - v - ax)) / self.norm(ax)
        
        return dist.mean()
        
    
    def ScalarToScalarLoss(self, s1, s2):
        
        '''
            s1: B x 1 or B 
            s2: B x 1 or B
        '''

        if s1.dim() == 1:
            return (s1 - s2).mean()
        
        return (s1-s2).squeeze(-1).mean()
    
    def PointToPlaneLoss(self, p, v, n):
        
        '''
            p: B x 3 query points
            n: B x 3 plane normals
            v: B x 3 points on the respective planes
        '''
        
        #n = n / self.norm(n)
        #d = -(n.unsqueeze(1) @ v.unsqueeze(-1)).squeeze(-1)
        #return (n.unsqueeze(1) @ p.unsqueeze(-1)).squeeze(-1) + d
        
        n = n / self.norm(n)
        d = (n.unsqueeze(1) @ (p - v).unsqueeze(-1)).squeeze(-1)
    
        return d.mean()
    
    def norm(self, v):
        
        '''
            returns the lengths of B vectors contained in a tensor
            B x d
        '''
        
        return (v*v).sum(-1).sqrt()