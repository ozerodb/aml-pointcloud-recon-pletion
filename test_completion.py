import argparse
import torch
import sys
from pathlib import Path
import random
import numpy as np

from our_modules import autoencoder, encoder, decoder, dataset, utils, cloud_utils
from our_modules.dataset import ShapeNetDataset, NovelCatDataset
from our_modules.loss import PointLoss
from our_modules.utils import bcolors
from models import dgpp

from pytorch_model_summary import summary as model_summary
from tqdm import tqdm

# quick example: 'python test_completion.py --encoder=dg --code_size=512 --decoder=ppd --checkpoint_path=latest'

def get_args():
    parser = argparse.ArgumentParser()
    # testing parameters
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--iterations', type=int, default=5)
    parser.add_argument('--checkpoint_path', type=str, required=True, help='the path of the checkpoint you want to test, can also be "latest"')
    parser.add_argument('--dataset', type=str, default='shapenet', choices=['shapenet','novel_similar','novel_dissim'], help='which dataset to use for testing')
    parser.add_argument('--save_significant', type=str, default='False', choices=['True','False'], help='whether to save best and worst pointclouds')
    parser.add_argument('--force_cpu', type=str, default='False', choices=['True','False'], help='whether to enforce using the cpu')

    # autoencoder parameters
    parser.add_argument('--model', type=str, default=None, choices=['dgpp'], help='which model to use, not required if you specify the other parameters')
    parser.add_argument('--encoder', type=str, default=None, choices=['pointnet', 'pointnetp1', 'dgcnn', 'dg'], help='which encoder to use, not required if you specified a model')
    parser.add_argument('--code_size', type=int, default=256, help='the size of the encoded feature vector, sudgested <= 512 or 1024 (no downsampling)')
    parser.add_argument('--decoder', type=str, default='fcm', choices=['fcs', 'fcm', 'fcl', 'ppd'], help='which decoder to use')

    # completion-specific parameters
    parser.add_argument('--remove_point_num', type=int, default=256, help='number of points to remove')
    parser.add_argument('--method', type=str, default='missing', choices=['total','missing'], help='whether the model should output total pointcloud or the missing patch only')

    return parser.parse_args()

def test(model, device, test_dataloader, method, remove_point_num):
    best_batch_loss = 999999
    worst_batch_loss = -1
    best_batch_index = 0
    worst_batch_index = 0

    chamfer_distance = PointLoss()
    model.eval()
    tot_loss = 0.0
    tot_samples = 0
    with torch.no_grad():
        for i, original_pointclouds in enumerate(test_dataloader):
            batch_size, _, _ = original_pointclouds.size()
            tot_samples += batch_size
            partials, removeds = [], []

            for m in range(batch_size): # for each pointcloud
                viewpoint = utils.pick_random_point_on_sphere() # pick a random point on the unitary sphere
                partial, removed = cloud_utils.torch_remove_closest_m_points_from_viewpoint(pointcloud=original_pointclouds[m], viewpoint=viewpoint, m=remove_point_num)   # drop the closest m points

                partials.append(partial)
                removeds.append(removed)

            partials = torch.stack(partials).to(device)  # move the partial pointclouds to training device, as they will be used in any case
            reconstructed_pointclouds = model(partials)

            if method=="total":
                original_pointclouds = original_pointclouds.to(device)
                loss = chamfer_distance(reconstructed_pointclouds, original_pointclouds)    # CD between total shapes
            else:   # method == "missing"
                removeds = torch.stack(removeds).to(device)
                loss = chamfer_distance(reconstructed_pointclouds, removeds)    # CD between missing shapes only
            float_loss = loss.item()
            tot_loss += (float_loss * batch_size)

            if float_loss > worst_batch_loss:
                worst_batch_loss = float_loss
                worst_batch_index = i
            if float_loss < best_batch_loss:
                best_batch_loss = float_loss
                best_batch_index = i

    test_loss = (tot_loss*1.0/float(len(test_dataloader.dataset)))
    test_loss = round(test_loss, 4)
    return test_loss, best_batch_index, worst_batch_index

def main():
    args = get_args()
    device = 'cuda' if (torch.cuda.is_available() and not args.force_cpu=="True") else 'cpu'
    print('Testing on', device)

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
    print(model_summary(model, torch.zeros((args.batch_size, 1024, 3)).to(device)))

    # Loading the checkpoint
    if args.checkpoint_path == "latest":
        import os
        dir_name = './weights'
        list_of_files = sorted( filter( lambda x: os.path.isfile(os.path.join(dir_name, x)), os.listdir(dir_name) ) )
        args.checkpoint_path = "weights/"+list_of_files[-1]
    print("Selected checkpoint:", args.checkpoint_path)
    checkpoint = torch.load(args.checkpoint_path, map_location=torch.device(device))
    model.load_state_dict(checkpoint['model_state_dict'])

    # Loading the datasets, class by class, and testing
    if args.dataset == 'shapenet':
        print('Testing completion on ShapeNet dataset')
        categories = ['Airplane', 'Chair', 'Table', 'Lamp', 'Car', 'Motorbike', 'Mug']
    elif args.dataset == 'novel_similar':
        print('Testing completion on similar Novel Categories dataset')
        categories = ['basket', 'bicycle', 'bowl', 'helmet', 'microphone', 'rifle', 'watercraft']
    elif args.dataset == 'novel_dissim':
        print('Testing completion on dissimilar Novel Categories dataset')
        categories = ['bookshelf', 'bottle', 'clock', 'microwave', 'pianoforte', 'telephone']
    try:
        for cat in categories:
            if args.dataset == 'shapenet':
                test_dataset = ShapeNetDataset( 'dataset_shapenet',
                        npoints=1024,
                        classification=False,
                        class_choice=cat,
                        split='test',
                        data_augmentation=False)
            elif args.dataset == 'novel_similar':
                test_dataset = NovelCatDataset('dataset_novel',
                        npoints=1024,
                        cat='similar',
                        class_choice=cat)
            elif args.dataset == 'novel_dissim':
                test_dataset = NovelCatDataset('dataset_novel',
                        npoints=1024,
                        cat='dissim',
                        class_choice=cat)

            test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
            bests = []
            worsts = []
            final_cat_test_loss = 0
            for i in tqdm(range(args.iterations)):
                it_cat_test_loss, best_batch_index, worst_batch_index = test(model, device, test_dataloader, args.method, args.remove_point_num)
                final_cat_test_loss += it_cat_test_loss
                bests.append(best_batch_index)
                worsts.append(worst_batch_index)

            final_cat_test_loss /= args.iterations
            print(f"Category: '{cat}' - Number of samples in test dataset: {len(test_dataset)} - Average test loss over {args.iterations} iterations: {final_cat_test_loss:.04f}" )

            save_significant = True if args.save_significant == "True" else False
            if args.save_significant:
                # save best batch of pointclouds
                best_batch_index = random.choice(bests)
                starting_index = best_batch_index * args.batch_size
                best_path = f'./significant_pointclouds/{cat}/best_batch'
                Path(best_path).mkdir(parents=True, exist_ok=True)
                for i in range(starting_index, starting_index+args.batch_size):
                    try:
                        pc = test_dataset.__getitem__(i)
                        np.savetxt(f"{best_path}/{i-starting_index}.pts", pc.cpu().detach().numpy())
                    except IndexError:
                        break
                # save worst batch of pointclouds
                worst_batch_index = random.choice(worsts)
                starting_index = worst_batch_index * args.batch_size
                worst_path = f'./significant_pointclouds/{cat}/worst_batch'
                Path(worst_path).mkdir(parents=True, exist_ok=True)
                for i in range(starting_index, starting_index+args.batch_size):
                    try:
                        pc = test_dataset.__getitem__(i)
                        np.savetxt(f"{worst_path}/{i-starting_index}.pts", pc.cpu().detach().numpy())
                    except IndexError:
                        break

    except KeyboardInterrupt:
        pass
if __name__ == '__main__':
    main()