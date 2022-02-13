import argparse
import torch
import sys

from our_modules import autoencoder, encoder, decoder, dataset, utils, cloud_utils
from our_modules.dataset import ShapeNetDataset, NovelCatDataset
from our_modules.loss import PointLoss
from our_modules.utils import bcolors
from models import dgpp

from pytorch_model_summary import summary as model_summary
from tqdm import tqdm

# quick example: 'python test_reconstruction.py --encoder=dgcnn --code_size=256 --decoder=ppd --checkpoint_path=latest'

def get_args():
    parser = argparse.ArgumentParser()
    # testing parameters
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--iterations', type=int, default=5)
    parser.add_argument('--checkpoint_path', type=str, required=True, help='the path of the checkpoint you want to test, can also be "latest"')
    parser.add_argument('--dataset', type=str, default='shapenet', choices=['shapenet','novel_similar','novel_dissim'], help='which dataset to use for testing')

    # autoencoder parameters
    parser.add_argument('--model', type=str, default=None, choices=['dgpp'], help='which model to use, not required if you specify the other parameters')
    parser.add_argument('--encoder', type=str, default=None, choices=['pointnet', 'pointnetp1', 'dgcnn', 'dg'], help='which encoder to use, not required if you specified a model')
    parser.add_argument('--code_size', type=int, default=256, help='the size of the encoded feature vector, sudgested <= 512 or 1024 (no downsampling)')
    parser.add_argument('--decoder', type=str, default='fcm', choices=['fcs', 'fcm', 'fcl', 'ppd'], help='which decoder to use')

    return parser.parse_args()

def test(model, device, test_dataloader, method):
    chamfer_distance = PointLoss()
    model.eval()
    tot_loss = 0.0
    tot_samples = 0
    with torch.no_grad():
        for original_pointclouds in test_dataloader:
            batch_size, _, _ = original_pointclouds.size()
            tot_samples += batch_size
            original_pointclouds = original_pointclouds.to(device)

            reconstructed_pointclouds = model(original_pointclouds)
            loss = chamfer_distance(reconstructed_pointclouds, original_pointclouds)    # CD between total shapes

            tot_loss += (loss.item() * batch_size)

    test_loss = (tot_loss*1.0/float(len(test_dataloader.dataset)))
    test_loss = round(test_loss, 4)
    return test_loss

def main():
    args = get_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Testing on', device)

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
        print('Testing reconstruction on ShapeNet dataset')
        categories = ['Airplane', 'Chair', 'Table', 'Lamp', 'Car', 'Motorbike', 'Mug']
    elif args.dataset == 'novel_similar':
        print('Testing reconstruction on similar Novel Categories dataset')
        categories = ['basket', 'bicycle', 'bowl', 'helmet', 'microphone', 'rifle', 'watercraft']
    elif args.dataset == 'novel_dissim':
        print('Testing reconstruction on dissimilar Novel Categories dataset')
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

            final_cat_test_loss = 0
            for i in tqdm(range(args.iterations)):
                final_cat_test_loss += test(model, device, test_dataloader)
            final_cat_test_loss /= args.iterations

            print(f"Category: '{cat}' - Number of samples in test dataset: {len(test_dataset)} - Average test loss over {args.iterations} iterations: {final_cat_test_loss:.04f}" )

    except KeyboardInterrupt:
        pass
if __name__ == '__main__':
    main()