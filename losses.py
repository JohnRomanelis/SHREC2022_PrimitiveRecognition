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
        
        #torch Tensor: B
        return  (torch.ones(B).to(ax1.device) - (ax1.unsqueeze(1) @ ax2.unsqueeze(-1)).abs().squeeze(-1))#.mean()

    def AxisToAxisAngle(self, ax1, ax2):
        '''
            ax1: Bx3
            ax2: Bx3
        '''
        
        B = ax1.shape[0]
        
        #normalizing
        ax1 = ax1 / self.norm(ax1)
        ax2 = ax2 / self.norm(ax2)

        #torch Tensor: B
        return (ax1.unsqueeze(1) @ ax2.unsqueeze(-1)).acos().squeeze(-1)#.mean()


    def L1Loss(self, ax1, ax2):

        '''
            ax1: B x 3
            ax2: B x 3
        '''
        
        #torch Tensor: B
        return (ax1 - ax2).abs().sum(-1)#.mean()


    def PointToPointLoss(self, p1, p2):
        
        '''
            p1: Bx3
            p2: Bx3
        '''
        q = p1 - p2

        #torch Tensor: B
        return (q * q).sum(-1)#.mean()
    
    def PointToAxisLoss(self, p, ax, v):
        
        '''
            p:  Bx3 predicted point
            ax: Bx3 actual axis
            v:  Bx3 a point on the actual axis
        '''
        
        dist = self.norm(torch.cross(p-v, p - v - ax)) / self.norm(ax)
        
        #torch Tensor: B
        return dist#.mean()
        
    
    def ScalarToScalarLoss(self, s1, s2):
        
        '''
            s1: B x 1 or B 
            s2: B x 1 or B
        '''

        if s1.dim() == 1:
            return ((s1-s2) * (s1 - s2)).mean()
        
        #torch Tensor: B
        return ((s1-s2) * (s1-s2)).squeeze(-1)#.mean()
    
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
    
        #torch Tensor: B
        return d.abs()#.mean()
    
    def norm(self, v):
        
        ''' λύ
            returns the lengths of B vectors contained in a tensor
            B x d
        '''
        
        return (v*v).sum(-1, keepdim=True).sqrt()


class PlaneLoss(Losses):
    
    def __call__(self, plane_pred, actual_plane, trans):
        
        pred_normal, pred_vertex = self.transform_plane_outputs(plane_pred, trans)

        actual_normal, actual_vertex = actual_plane[:,:3], actual_plane[:,3:6]
        
        #
        n_loss = self.AxisToAxisLoss(pred_normal, actual_normal)
        #
        v_loss = self.PointToPlaneLoss(pred_vertex, actual_vertex, actual_normal)
        
        return n_loss + v_loss
    

    def transform_plane_outputs(self, plane_pred, trans):
        
        #plane_pred: B x 6
        #scale: B x 1
        #shift: B x 3
        #rotation_mat: B x 3 x 3

        scale, shift, rotation_mat = trans["norm_factors"], trans["shifts"], trans["inv_rotations"]
        
        scale = scale.to(plane_pred.device).unsqueeze(-1)
        shift = shift.to(plane_pred.device)
        
        normal = plane_pred[:,0:3]
        vertex = plane_pred[:,3:]
        if rotation_mat is not None:
            rotation_mat = rotation_mat.to(plane_pred.device)
            #applying inverse rotation to normal vectors
            normal = (rotation_mat @ normal.unsqueeze(-1)).squeeze(-1)
        #applying inverse rotation, scaling and translation to vertices
            vertex = (rotation_mat @ vertex.unsqueeze(-1)).squeeze(-1)
        vertex = vertex * scale
        vertex = vertex - shift
        
        return normal, vertex
        
class CylinderLoss(Losses):
    

    def __call__(self, cyl_pred, actual_cyl, trans):
        
        pred_r, pred_axis, pred_vertex = self.transform_cylinder_outputs(cyl_pred, trans)

        actual_r, actual_axis, actual_vertex = actual_cyl[:,0], actual_cyl[:,1:4], actual_cyl[:,4:7]
        
        #
        a_loss = self.AxisToAxisLoss(pred_axis, actual_axis)
        #
        v_loss = self.PointToAxisLoss(pred_vertex, actual_axis)
        #
        r_loss = self.ScalarToScalarLoss(pred_r, actual_r)
        
        return a_loss + v_loss + r_loss
    

    def transform_cylinder_outputs(self, cyl_pred, trans):
        
        #cyl_pred: B x 7
        #scale: B x 1
        #shift: B x 3
        #rotation_mat: B x 3 x 3

        scale, shift, rotation_mat = trans["norm_factors"], trans["shifts"], trans["inv_rotations"]

        scale = scale.to(cyl_pred.device).unsqueeze(-1)
        shift = shift.to(cyl_pred.device)
        
        r = cyl_pred[:,0]
        axis = cyl_pred[:,1:4]
        vertex = cyl_pred[:,4:7]

        if rotation_mat is not None:
            rotation_mat = rotation_mat.to(cyl_pred.device)
            #applying inverse rotation to normal vectors
            axis = (rotation_mat @ axis.unsqueeze(-1)).squeeze(-1)
        #applying inverse rotation, scaling and translation to vertices
            vertex = (rotation_mat @ vertex.unsqueeze(-1)).squeeze(-1)
        vertex = vertex * scale
        vertex = vertex - shift
        r = r * scale

        return r, axis, vertex

    
class ConeLoss(Losses):
    

    def __call__(self, cone_pred, actual_cone, trans):
        
        pred_theta, pred_axis, pred_vertex = self.transform_cone_outputs(cone_pred, trans)

        actual_theta, actual_axis, actual_vertex =  actual_cone[:,0], actual_cone[:,1:4], actual_cone[:,4:7]
        
        #
        a_loss = self.AxisToAxisLoss(pred_axis, actual_axis)
        #
        v_loss = self.PointToAxisLoss(pred_vertex, actual_axis)
        #
        t_loss = self.ScalarToScalarLoss(pred_theta, actual_theta)
        
        return a_loss + v_loss + t_loss
    

    def transform_cone_outputs(self, cone_pred, trans):
        
        #cone_pred: B x 7
        #scale: B x 1
        #shift: B x 3
        #rotation_mat: B x 3 x 3

        scale, shift, rotation_mat = trans["norm_factors"], trans["shifts"], trans["inv_rotations"]

        scale = scale.to(cone_pred.device)
        shift = shift.to(cone_pred.device)
        
        theta = cone_pred[:,0]
        axis = cone_pred[:,1:4]
        vertex = cone_pred[:,4:7]

        if rotation_mat is not None:
            rotation_mat = rotation_mat.to(cone_pred.device)
            #applying inverse rotation to normal vectors
            axis = (rotation_mat @ axis.unsqueeze(-1)).squeeze(-1)
        #applying inverse rotation, scaling and translation to vertices
            vertex = (rotation_mat @ vertex.unsqueeze(-1)).squeeze(-1)
        vertex = vertex * scale
        vertex = vertex - shift

        return theta, axis, vertex


class SphereLoss(Losses):
    

    def __call__(self, sphere_pred, actual_sphere, trans):
        
        pred_r, pred_center = self.transform_sphere_outputs(sphere_pred, trans)

        actual_r, actual_center = actual_sphere[:,0], actual_sphere[:,1:4] 
        
        c_loss = self.PointToPointLoss(pred_center, actual_center)
        r_loss = self.ScalarToScalarLoss(pred_r, actual_r)
        
        return c_loss + r_loss
    

    def transform_sphere_outputs(self, sphere_pred, trans):
        
        #cyl_pred: B x 7
        #scale: B x 1
        #shift: B x 3
        #rotation_mat: B x 3 x 3

        scale, shift, rotation_mat = trans["norm_factors"], trans["shifts"], trans["inv_rotations"]

        scale = scale.to(sphere_pred.device).unsqueeze(-1)
        shift = shift.to(sphere_pred.device)
        
        r = sphere_pred[:,0]
        center = sphere_pred[:,1:4]
        
        if rotation_mat is not None:
            rotation_mat = rotation_mat.to(sphere_pred.device)
        #applying inverse rotation, scaling and translation to vertices
            center = (rotation_mat @ center.unsqueeze(-1)).squeeze(-1)
        center = center * scale
        center = center - shift
        r = r * scale

        return r, center

    
class TorusLoss(Losses):
    
    
    def __call__(self, torus_pred, actual_torus, trans):
        
        pred_R, pred_r, pred_axis, pred_center = self.transform_torus_outputs(torus_pred, trans)

        actual_R, actual_r, actual_axis, actual_center = actual_torus[:,0], actual_torus[:,1], \
                                                         actual_torus[:,2:5], actual_torus[:,5:8]
        
        
        a_loss = self.AxisToAxisLoss(pred_axis, actual_axis)
        c_loss = self.PointToPointLoss(pred_center, actual_center)
        R_loss = self.ScalarToScalarLoss(pred_R, actual_R)
        r_loss = self.ScalarToScalarLoss(pred_r, actual_r)
        
        
        return a_loss + c_loss + R_loss + r_loss
    

    def transform_torus_outputs(self, torus_pred, trans):
        

        #cyl_pred: B x 7
        #scale: B x 1
        #shift: B x 3
        #rotation_mat: B x 3 x 3

        scale, shift, rotation_mat = trans["norm_factors"], trans["shifts"], trans["inv_rotations"]

        scale = scale.to(torus_pred.device).unsqueeze(-1)
        shift = shift.to(torus_pred.device)
        
        R = torus_pred[:,0]
        r = torus_pred[:,1]
        axis = torus_pred[:,2:5]
        center = torus_pred[:,5:]

        if rotation_mat is not None:
            rotation_mat = rotation_mat.to(torus_pred.device)
            #applying inverse rotation to normal vectors
            axis = (rotation_mat @ axis.unsqueeze(-1)).squeeze(-1)
        #applying inverse rotation, scaling and translation to vertices
            center = (rotation_mat @ center.unsqueeze(-1)).squeeze(-1)
        center = center * scale
        center = center - shift
        r = r * scale

        return R, r, axis, center