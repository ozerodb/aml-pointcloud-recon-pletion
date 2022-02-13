import argparse
import torch
import sys

import numpy as np

from our_modules import autoencoder, encoder, decoder, dataset, utils, cloud_utils, visualization
from our_modules.dataset import ShapeNetDataset, NovelCatDataset
from our_modules.loss import PointLoss
from our_modules.utils import bcolors
from models import dgpp

from pytorch_model_summary import summary as model_summary
from tqdm import tqdm

# quick example: 'python test_completion.py --encoder=dg --code_size=512 --decoder=ppd --checkpoint_path=latest'

def get_args():
    parser = argparse.ArgumentParser()
    # visualization parameters
    parser.add_argument('--checkpoint_path', type=str, required=True, help='the path of the checkpoint you want to use, can also be "latest"')
    parser.add_argument('--pointcloud_path', type=str, required=True, help='the path of the pointcloud you want to visualize')

    # autoencoder parameters
    parser.add_argument('--model', type=str, default=None, choices=['dgpp'], help='which model to use, not required if you specify the other parameters')
    parser.add_argument('--encoder', type=str, default=None, choices=['pointnet', 'pointnetp1', 'dgcnn', 'dg'], help='which encoder to use, not required if you specified a model')
    parser.add_argument('--code_size', type=int, default=256, help='the size of the encoded feature vector, sudgested <= 512 or 1024 (no downsampling)')
    parser.add_argument('--decoder', type=str, default='fcm', choices=['fcs', 'fcm', 'fcl', 'ppd'], help='which decoder to use')

    # completion-specific parameters
    parser.add_argument('--remove_point_num', type=int, default=256, help='number of points to remove')
    parser.add_argument('--method', type=str, default='missing', choices=['missing'], help='whether the model should output total pointcloud or the missing patch only')

    return parser.parse_args()

def main():
    args = get_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.model is not None:
            if args.encoder is not None:
                print("You mistakenly specified both the model and the encoder, I'll ignore the encoder.")
            if args.model == 'dgpp':
                if args.method == 'total':
                    print("dgpp is not (necessarily) meant for completion of the total shape, if you wish to try it anyway, consider specifiying the following parameters instead:")
                    print("--encoder=dg --code_size=512 --decoder=ppd")
                    sys.exit()
                model = dgpp.dgpp(remove_point_num=args.remove_point_num).to(device)
    else:
        model = autoencoder.build_model(enc_type=args.encoder,
                                    encoding_length=args.code_size,
                                    dec_type=args.decoder,
                                    method=args.method,
                                    remove_point_num=args.remove_point_num).to(device)
    print(model_summary(model, torch.zeros((16, 1024, 3)).to(device)))

    # Loading the checkpoint
    if args.checkpoint_path == "latest":
        import os
        dir_name = './weights'
        list_of_files = sorted( filter( lambda x: os.path.isfile(os.path.join(dir_name, x)), os.listdir(dir_name) ) )
        args.checkpoint_path = "weights/"+list_of_files[-1]
    print("Selected checkpoint:", args.checkpoint_path)
    checkpoint = torch.load(args.checkpoint_path, map_location=torch.device(device))
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.eval().to(device)
    # Loading the pointcloud
    pointcloud = np.loadtxt(args.pointcloud_path).astype(np.float32)
    pointcloud = torch.from_numpy(pointcloud)
    #original_pointcloud_o3d = o3d_pointcloud_from_torch(pointcloud)

    # preparing multiple visualizations of different croppings
    viewpoints = [torch.tensor([1,0,0]), torch.tensor([0,0,1]), torch.tensor([0,0,-1])]
    geometries = []
    for i, viewpoint in enumerate(viewpoints):
        # crop the pc
        partial_pointcloud, _ = cloud_utils.torch_remove_closest_m_points_from_viewpoint(pointcloud, viewpoint=viewpoint, m=args.remove_point_num)
        partial_pointcloud = partial_pointcloud[np.newaxis, ...].to(device)

        # infer the missing patch
        reconstructed_missing_patch = model(partial_pointcloud)[0]
        # prepare for visualization
        original_pointcloud_o3d = visualization.o3d_pointcloud_from_torch(pointcloud)
        partial_pointcloud_o3d = visualization.o3d_pointcloud_from_torch(partial_pointcloud[0])
        partial_pointcloud_o3d_2 = visualization.o3d_pointcloud_from_torch(partial_pointcloud[0])
        reconstructed_missing_patch_o3d = visualization.o3d_pointcloud_from_torch(reconstructed_missing_patch)
        visualization.o3d_paint_pointcloud_or_spherecloud_str(original_pointcloud_o3d, "red")
        visualization.o3d_paint_pointcloud_or_spherecloud_str(partial_pointcloud_o3d, "gray")
        visualization.o3d_paint_pointcloud_or_spherecloud_str(partial_pointcloud_o3d_2, "gray")
        visualization.o3d_paint_pointcloud_or_spherecloud_str(reconstructed_missing_patch_o3d, "green")
        R = partial_pointcloud_o3d.get_rotation_matrix_from_xyz((0, - np.pi / 2, 0))
        partial_pointcloud_o3d.rotate(R, center=(0, 0, 0))
        partial_pointcloud_o3d_2.rotate(R, center=(0, 0, 0))
        reconstructed_missing_patch_o3d.rotate(R, center=(0, 0, 0))
        partial_pointcloud_o3d.translate((0, -2*i, 0))
        partial_pointcloud_o3d_2.translate((2, -2*i, 0))
        original_pointcloud_o3d.rotate(R, center=(0, 0, 0))
        original_pointcloud_o3d.translate((4, -2*i, 0))
        reconstructed_missing_patch_o3d.translate((2, -2*i, 0))
        geometries.extend([partial_pointcloud_o3d, partial_pointcloud_o3d_2, reconstructed_missing_patch_o3d])
    visualization.o3d_visualize_geometries(geometries, "Point cloud completion")

if __name__ == '__main__':
    main()