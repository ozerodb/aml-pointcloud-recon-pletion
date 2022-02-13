from our_modules import visualization
import numpy as np
import sys
import argparse

# example: 'python3 visualize_pointcloud.py --pointcloud_path=dataset_shapenet/02691156/points/1a04e3eab45ca15dd86060f189eb133.pts'
# example: 'python3 visualize_pointcloud.py --pointcloud_path=dataset_shapenet/02691156/points/1a04e3eab45ca15dd86060f189eb133.pts --style=sphere'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pointcloud_path', type=str, required=True, help='the path of the pointcloud you want to visualize')
    parser.add_argument('--style', type=str, default='point', choices=['point', 'sphere'], help='whether to render points or spheres')
    args = parser.parse_args()

    np_pointcloud = np.loadtxt(args.pointcloud_path).astype(np.float32)
    o3d_pointcloud = visualization.o3d_pointcloud_from_numpy(np_pointcloud)
    if args.style == 'point':
        geometry = o3d_pointcloud
    else:
        geometry = visualization.o3d_pointcloud_to_spherecloud(o3d_pointcloud)

    visualization.o3d_paint_pointcloud_or_spherecloud_str(geometry, "blue") # 'red','light_red','green','light_green','blue','light_blue','black','gray','orange','yellow'
    visualization.o3d_visualize_geometries([geometry])

if __name__ == '__main__':
    main()