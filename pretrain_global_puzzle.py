import argparse
import torch
import torch.optim as optim
from our_modules import autoencoder, encoder, decoder, dataset, utils, visualization, cloud_utils
import open3d as o3d
from our_modules.dataset import ShapeNetDataset
from our_modules.utils import bcolors
import numpy as np
import torch.nn as nn
import time
from datetime import datetime
from tqdm import tqdm

class modified_DGCCN_encoder(nn.Module):
    def __init__(self, emb_dims=1024):
        super(modified_DGCCN_encoder, self).__init__()
        self.encoder = encoder.DGCNN_encoder(pooling_type=None)
        self.conv6 = nn.Sequential(
                            nn.Conv1d(1024, 128, kernel_size=1, bias=False),
                            nn.BatchNorm1d(128),
                            nn.ReLU()
                        )
        self.conv7 = nn.Conv1d(128, 8, kernel_size=1)

    def forward(self, points):
        x = self.encoder(points)   # [B, emb_dims, num_points]
        x = self.conv6(x) # [B, 128, num_points]

        x = self.conv7(x) # [B, 8, num_points]
        x = x.transpose(2,1) # [B, num_points, 8]
        return x

def get_args():
    parser = argparse.ArgumentParser()
    # training parameters
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--resume_checkpoint_path', type=str, default=None, help='if specified, resume training from this checkpoint')
    parser.add_argument('--data_augmentation', type=str, default='False', choices=['True','False'], help='whether to augment pointclouds during training')
    parser.add_argument('--save_weights', type=str, default='True', choices=['True','False'], help='whether to save the weights of the trained model')
    parser.add_argument('--force_cpu', type=str, default='False', choices=['True','False'], help='whether to enforce using the cpu')

    return parser.parse_args()

def main():
    args = get_args()
    use_cuda = True if (torch.cuda.is_available() and not args.force_cpu=="True") else False

    if use_cuda:    # look for the gpu with the lowest memory usage and select  it
        lowest_memory_usage_index = 0
        for i in range(torch.cuda.device_count()):
            lowest_memory_usage_index = i if torch.cuda.memory_reserved(i) < torch.cuda.memory_reserved(lowest_memory_usage_index) else lowest_memory_usage_index
        device = 'cuda:'+str(i)
    else:
        device = 'cpu'
    print('Training on', device)
    random_pc = torch.rand((16, 3, 1024)).to(device)
    model = modified_DGCCN_encoder().to(device)

    #target = target.view(-1, 1)[:, 0] - 1
    #print(target.size())

    # Loading the datasets
    categories = ['Airplane', 'Chair', 'Table', 'Lamp', 'Car', 'Motorbike', 'Mug']
    #categories = ['Mug']
    train_data_augmentation = True if args.data_augmentation == "True" else False

    train_dataset = ShapeNetDataset( 'dataset_shapenet',
                    npoints=1024,
                    classification=False,
                    class_choice=categories,
                    split='train',
                    data_augmentation=train_data_augmentation,
                    puzzle_segmentation=True)
    val_dataset = ShapeNetDataset( 'dataset_shapenet',
                    npoints=1024,
                    classification=False,
                    class_choice=categories,
                    split='val',
                    data_augmentation=False,
                    puzzle_segmentation=True)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True, drop_last=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=16, shuffle=False)

    print('Number of samples in training dataset:', len(train_dataset))
    print('Number of samples in validation dataset:', len(val_dataset))

    # Optimizer and loss criterion
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()
    # Loading a previous checkpoint
    if args.resume_checkpoint_path is not None:
        print("Resuming from checkpoint:", args.resume_checkpoint_path)
        checkpoint = torch.load(args.resume_checkpoint_path, map_location=torch.device(device))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict']) # this will override the lr

    # Training loop
    model = model.to(device)
    train_losses = []
    valid_losses = []
    date_str = datetime.now().strftime("%Y_%m_%d_%H_%M_%S") # YY_mm_dd_H_M_S - will be used for filenames
    print(f'## Starting global puzzle pretraining with id {date_str} ##')
    best_train_epoch = 0
    best_val_epoch = 0
    start_time = time.time()

    try:
        for epoch in range(args.epochs):  # loop over the dataset multiple times
            epoch_start_time = time.time()

            # Train
            model.train()
            tot_loss = 0.0
            tot_samples = 0
            for data in tqdm(train_dataloader):
                original_pc, scrambled_pc, seg = data
                batch_size = scrambled_pc.size(0)
                tot_samples += batch_size
                scrambled_pc = scrambled_pc.permute(0,2,1)
                scrambled_pc = scrambled_pc.float()
                scrambled_pc = scrambled_pc.to(device)

                seg = seg.to(device)
                optimizer.zero_grad()

                pred_seg = model(scrambled_pc)
                loss = criterion(pred_seg, seg)
                loss.backward()
                optimizer.step()
                tot_loss += (loss.item() * batch_size)

            train_loss = (tot_loss*1.0/float(tot_samples))
            train_loss = round(train_loss, 4)
            if epoch == 0:
                print("Epoch %d train loss is: %f" % (epoch, train_loss))
            else:
                if train_losses[-1] <= train_loss:
                    print(f"Epoch {epoch} train loss is: {train_loss} {bcolors.LIGHT_RED}(+{train_loss-train_losses[-1]}){bcolors.ENDC}")
                else:
                    print(f"Epoch {epoch} train loss is: {train_loss} {bcolors.LIGHT_GREEN}({train_loss-train_losses[-1]}){bcolors.ENDC}")
            train_losses.append(train_loss)
            best_train_epoch = epoch if train_loss < train_losses[best_train_epoch] else best_train_epoch

            # if epoch == (args.epochs - 1):
            #     colors = ['red','green','blue','light_blue','black','gray','orange','yellow']
            #     #scrambled_pc, seg, pred_seg
            #     original_pc = original_pc[0].numpy()
            #     scrambled_pc = scrambled_pc[0].cpu().detach().numpy()
            #     scrambled_pc = scrambled_pc.transpose(1,0)
            #     seg = seg[0].cpu().detach().numpy()
            #     pred_seg = pred_seg[0].cpu().detach().numpy()
            #     original_colors = []
            #     predicted_colors = []
            #     reconstructed_pc = []
            #     for i in range(1024):
            #         current_location_id = cloud_utils.retrieve_voxel_id(scrambled_pc[i])
            #         point_original_seg = np.argmax(seg[i])
            #         point_predicted_seg = np.argmax(pred_seg[i])
            #         predicted_location_id = point_predicted_seg
            #         reconstructed_pc.append(cloud_utils.move_voxel(scrambled_pc[i], current_location_id, predicted_location_id))
            #         point_original_color = colors[point_original_seg]
            #         point_predicted_color = colors[point_predicted_seg]
            #         original_colors.append(visualization.str_to_normalized_rgb(point_original_color))
            #         predicted_colors.append(visualization.str_to_normalized_rgb(point_predicted_color))
            #     original_pcd = visualization.o3d_pointcloud_from_numpy(original_pc)
            #     original_pcd.colors = o3d.utility.Vector3dVector(np.array(original_colors))

            #     scrambled1_pcd = visualization.o3d_pointcloud_from_numpy(scrambled_pc)
            #     scrambled1_pcd.colors = o3d.utility.Vector3dVector(np.array(original_colors))
            #     scrambled1_pcd.translate((0, 0, -3))

            #     scrambled2_pcd = visualization.o3d_pointcloud_from_numpy(scrambled_pc)
            #     scrambled2_pcd.colors = o3d.utility.Vector3dVector(np.array(predicted_colors))
            #     scrambled2_pcd.translate((0, 0, -6))

            #     reconstructed_pc = np.array(reconstructed_pc)
            #     reconstructed_pcd = visualization.o3d_pointcloud_from_numpy(reconstructed_pc)
            #     reconstructed_pcd.colors = o3d.utility.Vector3dVector(np.array(predicted_colors))
            #     reconstructed_pcd.translate((0, 0, -9))

            #     visualization.o3d_visualize_geometries([original_pcd, scrambled1_pcd, scrambled2_pcd, reconstructed_pcd])
            # continue
            # Validate
            model.eval()
            tot_loss = 0.0
            tot_samples = 0
            with torch.no_grad():
                for data in val_dataloader:
                    _, scrambled_pc, seg = data
                    batch_size = scrambled_pc.size(0)
                    tot_samples += batch_size
                    scrambled_pc = scrambled_pc.permute(0,2,1)
                    scrambled_pc = scrambled_pc.float()
                    scrambled_pc = scrambled_pc.to(device)
                    seg = seg.to(device)
                    pred_seg = model(scrambled_pc)
                    loss = criterion(pred_seg, seg)
                    tot_loss += (loss.item() * batch_size)

            valid_loss = (tot_loss*1.0/float(len(val_dataloader.dataset)))
            valid_loss = round(valid_loss, 4)
            if epoch == 0:
                print("Epoch %d valid loss is: %f" % (epoch, valid_loss))
            else:
                if valid_losses[-1] <= valid_loss:
                    print(f"Epoch {epoch} valid loss is: {valid_loss} {bcolors.LIGHT_RED}(+{valid_loss-valid_losses[-1]}){bcolors.ENDC}")
                else:
                    print(f"Epoch {epoch} valid loss is: {valid_loss} {bcolors.LIGHT_GREEN}({valid_loss-valid_losses[-1]}){bcolors.ENDC}")
            valid_losses.append(valid_loss)
            best_val_epoch = epoch if valid_loss < valid_losses[best_val_epoch] else best_val_epoch

            # Compute elapsed time and other statistics
            epoch_elapsed_time = time.time() - epoch_start_time
            average_time_per_epoch = (time.time() - start_time) / (epoch+1)
            remaining_epochs = args.epochs - epoch - 1
            print('Epoch time elapsed:', utils.pretty_time_delta(epoch_elapsed_time), '- estimated time remaining:', utils.pretty_time_delta(average_time_per_epoch*remaining_epochs))

    except KeyboardInterrupt:
        print('\n# Manual early stopping #', end='')

    elapsed_time = time.time() - start_time
    print('\n## Finished pretraining in', utils.pretty_time_delta(elapsed_time), '##\n')

# If it was possible to train for at least one epoch
    if train_losses and valid_losses:
        # Printing some information regarding loss at different epochs
        print(f'Best training loss {train_losses[best_train_epoch]} at epoch {best_train_epoch}')
        print(f'Best validation loss {valid_losses[best_val_epoch]} at epoch {best_val_epoch}')

        # Saving weights if user had requested it
        if args.save_weights == 'True':
            weights_filename = 'weights/'+date_str+'_global_puzzle_checkpoint.pth'
            torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()
                        }, weights_filename) # load with model.load_state_dict(torch.load(PATH))
            print(f"Saved checkpoint weights at {weights_filename}")

            weights_filename = 'weights/'+date_str+'_global_puzzle_weights.pth'
            torch.save({'model_state_dict': model.encoder.state_dict()}, weights_filename) # load with model.load_state_dict(torch.load(PATH))
            print(f"Saved local-only weights at {weights_filename}")

    else:
        print("Training interrupted before the very first epoch, didn't save the weights nor the report")

if __name__ == '__main__':
    main()

