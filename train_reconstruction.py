import argparse
import torch
import torch.optim as optim
import sys

from our_modules import autoencoder, encoder, decoder, dataset, utils
from our_modules.dataset import ShapeNetDataset
from our_modules.loss import PointLoss
from our_modules.utils import bcolors
from models import dgpp

import time
from datetime import datetime
from pytorch_model_summary import summary as model_summary
from tqdm import tqdm

# quick example: 'python train_reconstruction.py --encoder=pointnetp1 --code_size=128 --decoder=fcs --epochs=5 --save_weights=False --save_report=False'

def get_args():
    parser = argparse.ArgumentParser()
    # training parameters
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--resume_checkpoint_path', type=str, default=None, help='if specified, resume training from this checkpoint')
    parser.add_argument('--data_augmentation', type=str, default='False', choices=['True','False'], help='whether to augment pointclouds during training')
    parser.add_argument('--plot_losses', type=str, default='False', choices=['True','False'], help='whether to plot losses once training is over')
    parser.add_argument('--save_weights', type=str, default='True', choices=['True','False'], help='whether to save the weights of the trained model')
    parser.add_argument('--save_report', type=str, default='True', choices=['True','False'], help='whether to save the report of the training run')

    # autoencoder parameters
    parser.add_argument('--model', type=str, default=None, choices=['dgpp'], help='which model to use, not required if you specify the other parameters')
    parser.add_argument('--encoder', type=str, default=None, choices=['pointnet', 'pointnetp1', 'dgcnn', 'dg'], help='which encoder to use, not required if you specified a model')
    parser.add_argument('--code_size', type=int, default=256, help='the size of the encoded feature vector, sudgested <= 512 or 1024 (no downsampling)')
    parser.add_argument('--decoder', type=str, default='fcm', choices=['fcs', 'fcm', 'fcl', 'ppd'], help='which decoder to use')

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
    print(model_summary(model, torch.zeros((args.batch_size, 1024, 3)).to(device)))

    # Loading the datasets
    categories = ['Airplane', 'Chair', 'Table', 'Lamp', 'Car', 'Motorbike', 'Mug']
    train_data_augmentation = True if args.data_augmentation == "True" else False

    train_dataset = ShapeNetDataset( 'dataset_shapenet',
                    npoints=1024,
                    classification=False,
                    class_choice=categories,
                    split='train',
                    data_augmentation=train_data_augmentation)
    val_dataset = ShapeNetDataset( 'dataset_shapenet',
                    npoints=1024,
                    classification=False,
                    class_choice=categories,
                    split='val',
                    data_augmentation=False)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    print('Number of samples in training dataset:', len(train_dataset))
    print('Number of samples in validation dataset:', len(val_dataset))

    # Optimizer and loss criterion
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
    chamfer_distance = PointLoss()

    # Loading a previous checkpoint
    if args.resume_checkpoint_path is not None:
        checkpoint = torch.load(args.resume_checkpoint_path, map_location=torch.device(device))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict']) # this will override the lr

    # Training loop
    model = model.to(device)
    train_losses = []
    valid_losses = []
    date_str = datetime.now().strftime("%Y_%m_%d_%H_%M_%S") # YY_mm_dd_H_M_S - will be used for filenames
    print(f'## Starting point cloud reconstruction training with id {date_str} ##')
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
            for original_pointclouds in tqdm(train_dataloader):
                batch_size, _, _ = original_pointclouds.size()
                tot_samples += batch_size
                original_pointclouds = original_pointclouds.to(device)
                optimizer.zero_grad()
                reconstructed_pointclouds = model(original_pointclouds)
                loss = chamfer_distance(reconstructed_pointclouds, original_pointclouds)    # CD between total shapes
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

            # Validate
            model.eval()
            tot_loss = 0.0
            tot_samples = 0
            with torch.no_grad():
                for original_pointclouds in val_dataloader:
                    batch_size, _, _ = original_pointclouds.size()
                    tot_samples += batch_size
                    original_pointclouds = original_pointclouds.to(device)
                    reconstructed_pointclouds = model(original_pointclouds)
                    loss = chamfer_distance(reconstructed_pointclouds, original_pointclouds)    # CD between total shapes
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

            # Compute elapsed time and other statistics
            epoch_elapsed_time = time.time() - epoch_start_time
            average_time_per_epoch = (time.time() - start_time) / (epoch+1)
            remaining_epochs = args.epochs - epoch - 1
            print('Epoch time elapsed:', utils.pretty_time_delta(epoch_elapsed_time), '- estimated time remaining:', utils.pretty_time_delta(average_time_per_epoch*remaining_epochs))
            best_val_epoch = epoch if valid_loss < valid_losses[best_val_epoch] else best_val_epoch
    except KeyboardInterrupt:
        print('\n# Manual early stopping #', end='')

    elapsed_time = time.time() - start_time
    print('\n## Finished training in', utils.pretty_time_delta(elapsed_time), '##\n')

# If it was possible to train for at least one epoch
    if train_losses and valid_losses:
        # Printing some information regarding loss at different epochs
        print(f'Best training loss {train_losses[best_train_epoch]} at epoch {best_train_epoch}')
        print(f'Best validation loss {valid_losses[best_val_epoch]} at epoch {best_val_epoch}')

        # Generating textual report if user had requested it, saving as much infos as possible
        if args.save_report == 'True':
            report_filename = 'training_reports/'+date_str+'_reconstruction_report.txt'
            report_data = {}
            report_data['id'] = date_str
            report_data['args'] = args
            report_data['device'] = device
            report_data['model'] = model
            report_data['train_losses'] = train_losses
            report_data['valid_losses'] = valid_losses
            utils.save_training_report(report_data, report_filename)

        # Saving weights if user had requested it
        if args.save_weights == 'True':
            weights_filename = 'weights/'+date_str+'_reconstruction_weights.pth'
            torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()
                        }, weights_filename) # load with model.load_state_dict(torch.load(PATH))
            print(f"Saved weights at {weights_filename}")

        # Plotting losses if user had requested it
        if args.plot_losses == 'True':
            utils.plot_losses(train_losses, valid_losses)
    else:
        print("Training interrupted before the very first epoch, didn't save the weights nor the report")

if __name__ == '__main__':
    main()