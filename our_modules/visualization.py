import numpy as np
import open3d as o3d
import torch


def o3d_pointcloud_from_numpy(np_pointcloud):
    pc = o3d.geometry.PointCloud()    # instantiate PointCloud object
    pc.points = o3d.utility.Vector3dVector(np_pointcloud)    # set the PointCloud's points to the ones we are interested in
    pc.paint_uniform_color([0.7, 0.7, 0.7])  # set only one color
    return pc

def o3d_pointcloud_from_torch(torch_pointcloud):
    np_pointcloud = torch_pointcloud.cpu().detach().numpy()
    return o3d_pointcloud_from_numpy(np_pointcloud)

def o3d_pointcloud_to_spherecloud(o3d_pointcloud):
    list_of_spheres = []
    points = np.asarray(o3d_pointcloud.points) # retrieve the points of the pointcloud

    for point in points: # for each point
        mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.005, resolution=6) # instantiate a mesh sphere
        mesh_sphere.vertices = o3d.utility.Vector3dVector(np.asarray(mesh_sphere.vertices) + point) # change the position of its vertices by adding the coordinates of the point's center
        list_of_spheres.append(mesh_sphere)

    return list_of_spheres

def flatten_list(_2d_list):
    flat_list = []
    # Iterate through the outer list
    for element in _2d_list:
        if type(element) is list:
            # If the element is of type list, iterate through the sublist
            for item in element:
                flat_list.append(item)
        else:
            flat_list.append(element)
    return flat_list

def o3d_visualize_geometries(geometry_list=[], window_name="Geometries"):
    flat_list = flatten_list(geometry_list)
    o3d.visualization.draw_geometries(geometry_list=flat_list, window_name=window_name)

def str_to_rgb(str_color):
    colors = {
        'red' : [255, 0, 0],
        'light_red' : [255, 50, 50],
        'green' : [0, 200, 0],
        'light_green' : [90, 215, 90],
        'blue' : [0, 0, 255],
        'light_blue' : [0, 100, 255],
        'black' : [0, 0, 0],
        'gray' : [180, 180, 180],
        'orange' : [255, 150, 50],
        'yellow' : [255, 220, 100]
    }
    return colors[str_color]

def normalize_rgb(rgb_color=[]):
    normalized = []
    for channel in rgb_color:
        normalized.append(channel / 255)
    return normalized

def str_to_normalized_rgb(str_color):
    colors = {
        'red' : [255, 0, 0],
        'light_red' : [255, 50, 50],
        'green' : [0, 200, 0],
        'light_green' : [90, 215, 90],
        'blue' : [0, 0, 255],
        'light_blue' : [0, 100, 255],
        'black' : [0, 0, 0],
        'gray' : [180, 180, 180],
        'orange' : [255, 150, 50],
        'yellow' : [255, 220, 100]
    }
    return normalize_rgb(colors[str_color])

def o3d_paint_pointcloud_or_spherecloud_rgb(geometry, rgb_color=[255, 0, 0]):
    normalized_rgb = normalize_rgb(rgb_color)

    if type(geometry) is o3d.geometry.PointCloud:
        geometry.paint_uniform_color(normalized_rgb)

    if type(geometry) is list:   # spherecloud
        for sphere in geometry:
            sphere.paint_uniform_color(normalized_rgb)

def o3d_paint_pointcloud_or_spherecloud_str(geometry, str_color="red"):
    rgb_color = str_to_rgb(str_color)
    o3d_paint_pointcloud_or_spherecloud_rgb(geometry, rgb_color)