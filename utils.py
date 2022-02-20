import MinkowskiEngine as ME
import torch
import open3d as o3d

def minkowski_collate(list_data):

    if list_data[0]['y'] is not None:
        coordinates, features, labels = ME.utils.sparse_collate(
            [d['x'] for d in list_data],
            [d['x'] for d in list_data],
            [d['y'].unsqueeze(0) for d in list_data],
            dtype = torch.float32
        )
    else:
        coordinates, features = ME.utils.sparse_collate(
            [d['x'] for d in list_data],
            [d['x'] for d in list_data],
            dtype = torch.float32
        )
        labels = None

    # collating other data
    norm_factors = []
    shifts = []
    inv_rotations = []
    means = []
    
    for d in list_data:
        norm_factors.append(d['norm_factor'])
        shifts.append(d['shift'])
        if 'inverse_rotation' in d.keys():
            inv_rotations.append(d['inverse_rotation'])
        means.append(d['mean'])
        
    norm_factors = torch.stack(norm_factors)
    shifts = torch.stack(shifts)
    
    if len(inv_rotations) > 0:
        inv_rotations = torch.stack(inv_rotations)
    
    means = torch.stack(means)
    
    ret = {
        "coordinates"   : coordinates, 
        "features"      : features,
        "labels"        : labels,
        "means"         : means,
        "trans": {"norm_factors"  : norm_factors,
                  "shifts"        : shifts,
                  "inv_rotations" : inv_rotations if len(inv_rotations) > 0 else None
                 }
        }


    if 'initial_points' in list_data[0].keys():
        initial_points = [d['initial_points'] for d in list_data]

    ret['initial_points'] = initial_points

    return ret

def create_input_batch(batch, device="cuda", quantization_size=0.05):
    batch["coordinates"][:, 1:] = batch["coordinates"][:, 1:] / quantization_size
    return ME.TensorField(
        coordinates=batch["coordinates"],
        features=batch["features"],
        device=device
    )


def makeO3Dpc(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.numpy())
    return pcd

def visualize_pointcloud(pcd):
    if not isinstance(pcd, o3d.geometry.PointCloud):
        pcd = makeO3Dpc(pcd)
    o3d.visualization.draw_geometries([pcd])