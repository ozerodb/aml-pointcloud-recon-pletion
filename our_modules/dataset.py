import torch.utils.data as data
import os
import os.path
import torch
import numpy as np
import json
import h5py
import random
from . import cloud_utils

############### SHAPENET DATASET ###############

def gen_modelnet_id(root):
    classes = []
    with open(os.path.join(root, 'train.txt'), 'r') as f:
        for line in f:
            classes.append(line.strip().split('/')[0])
    classes = np.unique(classes)
    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../misc/modelnet_id.txt'), 'w') as f:
        for i in range(len(classes)):
            f.write('{}\t{}\n'.format(classes[i], i))

class ShapeNetDataset(data.Dataset):
    def __init__(self,
                root,
                npoints=2048,
                classification=False,
                class_choice=None,
                split='train',
                data_augmentation=False,
                puzzle_segmentation=False,
                rotation_recognition=False):
        self.npoints = npoints
        self.root = root
        self.catfile = os.path.join(self.root, 'synsetoffset2category.txt')
        self.cat = {}
        self.data_augmentation = data_augmentation
        self.classification = classification
        self.puzzle_segmentation = puzzle_segmentation
        self.rotation_recognition = rotation_recognition

        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]

        if not class_choice is None:
            self.cat = {k: v for k, v in self.cat.items() if k in class_choice}

        self.id2cat = {v: k for k, v in self.cat.items()}

        self.meta = {}
        splitfile = os.path.join(self.root, 'train_test_split', 'shuffled_{}_file_list.json'.format(split))

        filelist = json.load(open(splitfile, 'r'))
        for item in self.cat:
            self.meta[item] = []

        for file in filelist:
            _, category, uuid = file.split('/')
            if category in self.cat.values():
                self.meta[self.id2cat[category]].append((os.path.join(self.root, category, 'points', uuid+'.pts'),
                                        os.path.join(self.root, category, 'points_label', uuid+'.seg')))

        self.datapath = []
        for item in self.cat:
            for fn in self.meta[item]:
                self.datapath.append((item, fn[0], fn[1]))

        self.classes = dict(zip(sorted(self.cat), range(len(self.cat))))
        #print(self.classes)

    def __getitem__(self, index):
        fn = self.datapath[index]
        cls = self.classes[self.datapath[index][0]]

        point_set = np.loadtxt(fn[1]).astype(np.float32)    # load the point cloud
        point_set = point_set[np.random.choice(point_set.shape[0], self.npoints, replace=True), :]    # resample
        point_set = cloud_utils.normalize_pointcloud(point_set) # center and normalize

        if self.data_augmentation:
            point_set = cloud_utils.random_rotate_pointcloud(point_set) # random rotation
            point_set = cloud_utils.random_jitter_pointcloud(point_set) # random jitter

        if self.classification:
            return torch.from_numpy(point_set), cls
        elif self.puzzle_segmentation:
            seg = []
            regions_reference_point = [np.array([1,1,1]),np.array([1,1,0]),np.array([1,0,1]),np.array([1,0,0]),np.array([0,1,1]),np.array([0,1,0]),np.array([0,0,1]),np.array([0,0,0])]
            original_references = []
            target_references = []
            scrambled_regions = [i for i in range(8)]
            random.shuffle(scrambled_regions)
            for point in point_set:     # please don't judge me for this mess, I didn't have much time
                x, y, z = point[0], point[1], point[2]
                if x>0:
                    if y>0:
                        if z>0:
                            original_region = 0   # x>0, y>0, z>0
                        else:
                            original_region = 1   # x>0, y>0, z<=0
                    else:
                        if z>0:
                            original_region = 2   # x>0, y<=0, z>0
                        else:
                            original_region = 3   # x>0, y<=0, z<=0
                else:
                    if y>0:
                        if z>0:
                            original_region = 4   # x<=0, y>0, z>0
                        else:
                            original_region = 5   # x<=0, y>0, z<=0
                    else:
                        if z>0:
                            original_region = 6   # x<=0, y<=0, z>0
                        else:
                            original_region = 7   # x<=0, y<=0, z<=0
                label = np.zeros(8)
                label[original_region] = 1
                seg.append(label)
                target_region = scrambled_regions[original_region]
                original_references.append(regions_reference_point[original_region])
                target_references.append(regions_reference_point[target_region])
            original_references = np.array(original_references)
            target_references = np.array(target_references)
            diff = target_references - original_references
            puzzled_point_set = point_set + diff
            seg = np.array(seg)
            seg = torch.from_numpy(seg)
            point_set = torch.from_numpy(point_set)
            puzzled_point_set = torch.from_numpy(puzzled_point_set)
            return point_set, puzzled_point_set, seg
        elif self.rotation_recognition:
            chosen_rotation = random.randint(0, 3)
            point_set = cloud_utils.rotate_pointcloud_x_axis(point_set, chosen_rotation)
            return torch.from_numpy(point_set), chosen_rotation
        else:   #puzzle segmentation
            return torch.from_numpy(point_set)
    def __len__(self):
        return len(self.datapath)

############### NOVEL CATEGORIES DATASET ###############

class NovelCatDataset(data.Dataset):
    def __init__(self, root, npoints=1024, cat='similar', class_choice=None):
        super().__init__()
        self.dataset = None
        self.npoints = npoints

        dirPath = os.path.join(root, 'sncore_v1_' + cat + 'Cat')

        if cat == 'similar':
            self.possible_cat = ['basket','bicycle','bowl','helmet','microphone','rifle','watercraft']
        else:
            self.possible_cat = ['bookshelf','bottle','clock','microwave','pianoforte','telephone']

        if class_choice is not None:
            self.cat = [v for v in self.possible_cat if v in class_choice]
        else:
            self.cat = self.possible_cat
        #print("Chosen categories:", self.cat)
        for _, _, files in os.walk(dirPath):
            for file in files:
                file_cat = file.split('_')[0]
                if file_cat in self.cat:
                    #print("loading category", file_cat)
                    with h5py.File(os.path.join(dirPath, file)) as f:
                        cat_dataset = np.array(f['points'])
                        self.dataset = np.concatenate((self.dataset, cat_dataset), axis=0) if self.dataset is not None else cat_dataset

    def __getitem__(self, index):
        point_set = self.dataset[index]
        point_set = point_set[np.random.choice(point_set.shape[0], self.npoints, replace=True), :]  # resample
        point_set = cloud_utils.normalize_pointcloud(point_set) # center and normalize

        return torch.from_numpy(point_set)

    def __len__(self):
        return len(self.dataset)
