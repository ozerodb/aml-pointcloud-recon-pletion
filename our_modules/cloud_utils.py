import numpy as np
import torch

def normalize_pointcloud(pointcloud):
    centroid = np.mean(pointcloud, axis=0)
    pointcloud = pointcloud - centroid
    dist = np.max(np.sqrt(np.sum(pointcloud ** 2, axis=1)))
    pointcloud = pointcloud / dist
    return pointcloud

def random_rotate_pointcloud(pointcloud):
    theta = np.random.uniform(0, np.pi*2)
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    pointcloud = pointcloud.dot(rotation_matrix)
    return pointcloud

def rotate_pointcloud_x_axis(pointcloud, rotation_factor):
    rotation_angle = np.pi*(rotation_factor/2)
    cosval = np.cos(rotation_angle)
    sinval = np.sin(rotation_angle)
    rotation_matrix = np.array([[1, 0, 0],
                                [0, cosval, -sinval],
                                [0, sinval, cosval]])
    pointcloud = pointcloud.dot(rotation_matrix)
    return pointcloud

def random_jitter_pointcloud(pointcloud, scale=0.02):
    pointcloud += np.random.normal(0, scale=scale, size=pointcloud.shape) # gaussian jitter
    return pointcloud

def torch_random_jitter_pointcloud(pointcloud, std=0.02):
    pointcloud += torch.normal(mean=0, std=std, size=pointcloud.shape) # gaussian jitter
    return pointcloud

def numpy_remove_closest_m_points_from_viewpoint(pointcloud, viewpoint=None, m=128):
    if viewpoint is None:
        viewpoint=np.array([1,0,0])
    diff = pointcloud - viewpoint   # compute the difference (not the distance) between each point of the pointcloud and the viewpoint

    dist = np.sqrt(np.sum(diff**2, axis=-1))     # compute the actual distance of each point from the viewpoint
    idx = np.argpartition(dist, m)  # partition the indices of the m closest points -> first m elements of idx are the indices of the closest points

    cropped_pointcloud = np.delete(pointcloud, idx[:m], axis=0) # remove the first m indices of idx from the pointcloud
    removed_points = pointcloud[idx[:m]]
    return cropped_pointcloud, removed_points

def torch_remove_closest_m_points_from_viewpoint(pointcloud, viewpoint=None, m=128):
    if viewpoint is None:
        viewpoint=torch.tensor([1,0,0])
    diff = pointcloud - viewpoint   # compute the difference (not the distance) between each point of the pointcloud and the viewpoint

    dist = torch.sqrt(torch.sum(diff**2, dim=-1))     # compute the actual distance of each point from the viewpoint
    values , indices = torch.topk(dist, m, largest=False, sorted=False)  # select the indices of the m closest points

    mask = torch.ones(pointcloud.size(dim=0), dtype=torch.bool)
    mask[indices] = False
    cropped_pointcloud = pointcloud[mask == True] # only select the kept points
    removed_points = pointcloud[mask == False] # only select the removed points
    return cropped_pointcloud, removed_points

def move_voxel(voxel, original_region, target_region):
    regions_reference_point = [np.array([1,1,1]),np.array([1,1,0]),np.array([1,0,1]),np.array([1,0,0]),np.array([0,1,1]),np.array([0,1,0]),np.array([0,0,1]),np.array([0,0,0])]
    original_region_reference = regions_reference_point[original_region]
    target_region_reference = regions_reference_point[target_region]
    diff = target_region_reference - original_region_reference
    moved_voxel = voxel + diff
    return moved_voxel

def retrieve_voxel_id(point):
    x, y, z = point[0], point[1], point[2]
    if x>0:
        if y>0:
            if z>0:
                voxel_id = 0   # x>0, y>0, z>0
            else:
                voxel_id = 1   # x>0, y>0, z<=0
        else:
            if z>0:
                voxel_id = 2   # x>0, y<=0, z>0
            else:
                voxel_id = 3   # x>0, y<=0, z<=0
    else:
        if y>0:
            if z>0:
                voxel_id = 4   # x<=0, y>0, z>0
            else:
                voxel_id = 5   # x<=0, y>0, z<=0
        else:
            if z>0:
                voxel_id = 6   # x<=0, y<=0, z>0
            else:
                voxel_id = 7   # x<=0, y<=0, z<=0
    return voxel_id