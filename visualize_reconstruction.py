import argparse
import torch
import sys
import numpy as np
from our_modules import autoencoder, visualization
from pytorch_model_summary import summary as model_summary

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

    return parser.parse_args()

def main():
    args = get_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.model is not None:
            if args.encoder is not None:
                print("You mistakenly specified both the model and the encoder, I'll ignore the encoder.")
            if args.model == 'dgpp':
                print("dgpp is not (necessarily) meant for reconstruction, if you wish to try it anyway, consider specifiying the following parameters instead:")
                print("--encoder=dg --code_size=512 --decoder=ppd")
                sys.exit()
    else:
        model = autoencoder.build_model(enc_type=args.encoder,
                                    encoding_length=args.code_size,
                                    dec_type=args.decoder,
                                    method='total').to(device)
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

    original_pointcloud = pointcloud[np.newaxis, ...].to(device)

    # infer the reconstructed pointcloud
    reconstructed_pointcloud = model(original_pointcloud)[0]

    # prepare for visualization
    original_pointcloud_o3d = visualization.o3d_pointcloud_from_torch(pointcloud)
    reconstructed_pointcloud_o3d = visualization.o3d_pointcloud_from_torch(reconstructed_pointcloud)
    visualization.o3d_paint_pointcloud_or_spherecloud_str(original_pointcloud_o3d, "blue")
    visualization.o3d_paint_pointcloud_or_spherecloud_str(reconstructed_pointcloud_o3d, "light_blue")
    R = original_pointcloud_o3d.get_rotation_matrix_from_xyz((0, - np.pi / 2, 0))
    original_pointcloud_o3d.rotate(R, center=(0, 0, 0))
    reconstructed_pointcloud_o3d.rotate(R, center=(0, 0, 0))
    reconstructed_pointcloud_o3d.translate((0, 0, -2))

    # visualize
    visualization.o3d_visualize_geometries([original_pointcloud_o3d, reconstructed_pointcloud_o3d], "Point cloud reconstruction")

if __name__ == '__main__':
    main()