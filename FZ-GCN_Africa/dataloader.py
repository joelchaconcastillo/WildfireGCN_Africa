import torch
import warnings
import pandas as pd
import numpy as np
import torch.utils.data
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.utils.data.sampler import Sampler
import torchvision.transforms as transforms

import random
from datetime import datetime
import json
import random
import shutil
from pathlib import Path
class ManualSampler(Sampler):
    def __init__(self, dataset):
        self.indices = list(range(len(dataset)))
    def __iter__(self):
        return iter(self.indices)
    def shuffle(self):
        random.shuffle(self.indices)
    def __len__(self):
        return len(self.indices)


def interpolate_missing_pixels(
        image: np.ndarray,
        mask: np.ndarray,
        method: str = 'nearest',
        fill_value: int = 0
):
    """
    :param image: a 2D image
    :param mask: a 2D boolean image, True indicates missing values
    :param method: interpolation method, one of
        'nearest', 'linear', 'cubic'.
    :param fill_value: which value to use for filling up data outside the
        convex hull of known pixel values.
        Default is 0, Has no effect for 'nearest'.
    :return: the image with missing values interpolated
    """
    from scipy import interpolate

    h, w = image.shape[:2]
    xx, yy = np.meshgrid(np.arange(w), np.arange(h))

    known_x = xx[~mask]
    known_y = yy[~mask]
    known_v = image[~mask]
    missing_x = xx[mask]
    missing_y = yy[mask]

    interp_values = interpolate.griddata(
        (known_x, known_y), known_v, (missing_x, missing_y),
        method=method, fill_value=fill_value
    )

    interp_image = image.copy()
    interp_image[missing_y, missing_x] = interp_values

    return interp_image

class FireDSDataModuleGCN:
    """
    A DataModule implements 5 key methods:
        - prepare_data (things to do on 1 GPU/TPU, not on every GPU/TPU in distributed mode)
        - setup (things to do on every accelerator in distributed mode)
        - train_dataloader (the training dataloader)
        - val_dataloader (the validation dataloader(s))
        - test_dataloader (the test dataloader(s))

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html
    """

    def __init__(
            self, args):
        super().__init__()
        self.args = args

        if not args.dataset_root:
            raise ValueError('dataset_root variable must be set. Check README')
       
        self.data_train = FireDataset_Graph_npy(dataset_root=args.dataset_root,
                                          train_val_test='train',
                                          features=args.features,
                                          nan_fill=args.nan_fill, args=args)
        self.data_val = FireDataset_Graph_npy(dataset_root=args.dataset_root, 
                                        train_val_test='val',
                                        features=args.features, 
                                        nan_fill=args.nan_fill, args=args)
        self.data_test1 = FireDataset_Graph_npy(dataset_root=args.dataset_root,
                                         train_val_test='test1',
                                         features=args.features,
                                         nan_fill=args.nan_fill, args=args)
        self.data_test2 = FireDataset_Graph_npy(dataset_root=args.dataset_root,
                                         train_val_test='test2',
                                         features=args.features,
                                         nan_fill=args.nan_fill, args=args)

    def train_dataloader(self):
        instanceManualSampler = ManualSampler(self.data_train)
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.args.minbatch_size,
            num_workers=self.args.num_workers,
            pin_memory=self.args.pin_memory,
            sampler=instanceManualSampler,
#            shuffle=True,
            #shuffle=False,
            prefetch_factor=self.args.prefetch_factor,
            persistent_workers=self.args.persistent_workers
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.args.minbatch_size,
            num_workers=self.args.num_workers,
            pin_memory=self.args.pin_memory,
            shuffle=False,
            prefetch_factor=self.args.prefetch_factor,
            persistent_workers=self.args.persistent_workers
        )

    def test_dataloader1(self):
        return DataLoader(
            dataset=self.data_test1,
            batch_size=self.args.minbatch_size,
            num_workers=self.args.num_workers,
            pin_memory=self.args.pin_memory,
            shuffle=False,
            prefetch_factor=self.args.prefetch_factor,
            persistent_workers=self.args.persistent_workers
        )
    def test_dataloader2(self):
        return DataLoader(
            dataset=self.data_test2,
            batch_size=self.args.minbatch_size,
            num_workers=self.args.num_workers,
            pin_memory=self.args.pin_memory,
            shuffle=False,
            prefetch_factor=self.args.prefetch_factor,
            persistent_workers=self.args.persistent_workers
        )



class FireDataset_Graph_npy(Dataset):
    def __init__(self, dataset_root: str = None,   
                 train_val_test: str = 'train', features: list = None,
                 nan_fill: float = -1., args = None):
        # dataset_root should be a str leading to the path where the data have been downloaded and decompressed
        # Make sure to follow the details in the readme for that
        dataset_root = Path(dataset_root)

        self.args = args
        self.features = features
        self.nan_fill = nan_fill
        dataset_path = dataset_root
        self.positives_list = list((dataset_path / 'Positives').glob('*'))
        self.positives_list = list(zip(self.positives_list, [1] * (len(self.positives_list))))
        val_year = 2018
        test_year1 = 2019
        test_year2 = 2020


#        self.train_positive_list = [(x, y, None) for (x, y) in self.positives_list if int(x.stem[:4]) < val_year]
#        self.val_positive_list = [(x, y, None) for (x, y) in self.positives_list if int(x.stem[:4]) >= val_year and int(x.stem[:4]) < test_year1]
        self.test1_positive_list = [(x, y, None) for (x, y) in self.positives_list if int(x.stem[:4]) == test_year1]
        self.test2_positive_list = [(x, y, None) for (x, y) in self.positives_list if int(x.stem[:4]) == test_year2]

        self.negatives_list = list((dataset_path / 'Negatives').glob('*'))
        self.negatives_list = list(zip(self.negatives_list, [0] * (len(self.negatives_list))))


#        self.train_negative_list = [(x, y, None) for (x, y) in self.negatives_list if int(x.stem[:4]) < val_year]
#        self.val_negative_list = [(x, y, None) for (x, y) in self.negatives_list if int(x.stem[:4]) >= val_year and int(x.stem[:4]) < test_year1]
        self.test1_negative_list = [(x, y, None) for (x, y) in self.negatives_list if int(x.stem[:4]) == test_year1]
        self.test2_negative_list = [(x, y, None) for (x, y) in self.negatives_list if int(x.stem[:4]) == test_year2]

###################################################################
        self.train_positive_list1 = [(x, y, None) for (x, y) in self.positives_list if int(x.stem[:4]) < val_year]
        self.train_positive_list2 = [(x, y, transforms.CenterCrop(5)) for (x, y) in self.positives_list if int(x.stem[:4]) < val_year]
        self.train_positive_list3 = [(x, y, None) for (x, y) in self.positives_list if int(x.stem[:4]) < val_year]

        self.train_negative_list1 = [(x, y, None) for (x, y) in self.negatives_list if int(x.stem[:4]) < val_year]
        self.train_negative_list2 = [(x, y, transforms.CenterCrop(5)) for (x, y) in self.negatives_list if int(x.stem[:4]) < val_year]
        self.train_negative_list3 = [(x, y, None) for (x, y) in self.negatives_list if int(x.stem[:4]) < val_year]

        for idx in range(len(self.train_positive_list3)):
          rotation = random.randint(-180, 180)
          self.train_positive_list3[idx] = (self.train_positive_list3[idx][0], self.train_positive_list3[idx][1], transforms.RandomAffine(degrees=(rotation,rotation)))

        for idx in range(len(self.train_negative_list3)):
          rotation = random.randint(-180, 180)
          self.train_negative_list3[idx] = (self.train_negative_list3[idx][0], self.train_negative_list3[idx][1], transforms.RandomAffine(degrees=(rotation,rotation)))


        self.val_positive_list1 = [(x, y, None) for (x, y) in self.positives_list if int(x.stem[:4]) == val_year]
        self.val_positive_list2 = [(x, y, transforms.CenterCrop(5)) for (x, y) in self.positives_list if int(x.stem[:4]) == val_year]
        self.val_positive_list3 = [(x, y, None) for (x, y) in self.positives_list if int(x.stem[:4]) == val_year]
#        self.val_positive_list4 = [(x, y, transforms.functional.hflip) for (x, y) in self.positives_list if int(x.stem[:4]) == val_year]
#        self.val_positive_list5 = [(x, y, transforms.functional.vflip) for (x, y) in self.positives_list if int(x.stem[:4]) == val_year]
#        self.val_positive_list6 = [(x, y, transforms.functional.equalize) for (x, y) in self.positives_list if int(x.stem[:4]) == val_year]

        self.val_negative_list1 = [(x, y, None) for (x, y) in self.negatives_list if int(x.stem[:4]) == val_year]
        self.val_negative_list2 = [(x, y, transforms.CenterCrop(5)) for (x, y) in self.negatives_list if int(x.stem[:4]) == val_year]
        self.val_negative_list3 = [(x, y, None) for (x, y) in self.negatives_list if int(x.stem[:4]) == val_year]
#        self.val_negative_list4 = [(x, y, transforms.functional.hflip) for (x, y) in self.negatives_list if int(x.stem[:4]) == val_year]
#        self.val_negative_list5 = [(x, y, transforms.functional.vflip) for (x, y) in self.negatives_list if int(x.stem[:4]) == val_year]
#        self.val_negative_list6 = [(x, y, transforms.functional.equalize) for (x, y) in self.negatives_list if int(x.stem[:4]) == val_year]

        for idx in range(len(self.val_positive_list3)):
          rotation = random.randint(-180, 180)
          self.val_positive_list3[idx] = (self.val_positive_list3[idx][0], self.val_positive_list3[idx][1], transforms.RandomAffine(degrees=(rotation,rotation)))


        for idx in range(len(self.val_negative_list3)):
          rotation = random.randint(-180, 180)
          self.val_negative_list3[idx] = (self.val_negative_list3[idx][0], self.val_negative_list3[idx][1], transforms.RandomAffine(degrees=(rotation,rotation)))

        self.val_positive_list = self.val_positive_list1 # + self.val_positive_list3 # + self.val_positive_list3 # + self.val_positive_list2 + self.val_positive_list3 # + self.val_positive_list4 + self.val_positive_list5 #+ self.val_positive_list6
        self.val_negative_list = self.val_negative_list1 # + self.val_negative_list3 #  + self.val_negative_list3 #+ self.val_negative_list2 + self.val_negative_list3 #+ self.val_negative_list4 + self.val_negative_list5 #+ self.val_negative_list6
        self.train_positive_list = self.train_positive_list1 # + self.train_positive_list3 # + self.train_positive_list3 #+ self.train_positive_list2 + self.train_positive_list3
        self.train_negative_list = self.train_negative_list1 # +self.train_negative_list3 # + self.train_negative_list3 #+ self.train_negative_list2 + self.train_negative_list3


        if train_val_test == 'train':
            print(f'Positives: {len(self.train_positive_list)} / Negatives: {len(self.train_negative_list)}')
            self.path_list = self.train_positive_list + self.train_negative_list
        elif train_val_test == 'val':
            print(f'Positives: {len(self.val_positive_list)} / Negatives: {len(self.val_negative_list)}')
            self.path_list = self.val_positive_list + self.val_negative_list
        elif train_val_test == 'test1':
            print(f'Positives: {len(self.test1_positive_list)} / Negatives: {len(self.test1_negative_list)}')
            self.path_list = self.test1_positive_list + self.test1_negative_list
        elif train_val_test == 'test2':
            print(f'Positives: {len(self.test2_positive_list)} / Negatives: {len(self.test2_negative_list)}')
            self.path_list = self.test2_positive_list + self.test2_negative_list


#        self.path_list = [( np.nan_to_num(pd.read_csv(x)[features].to_numpy().reshape(T, W, H, -1), nan=self.nan_fill), y, trans) for (x, y, trans) in self.path_list]

        print("Dataset length", len(self.path_list))
        random.shuffle(self.path_list)
        self.listZPI = []
        self.listData = []
        self.listLabels = []
        T, W, H = 10, 5, 5
        transRot = transforms.RandomAffine(degrees=(-180,180))
        for (path, label, trans) in self.path_list:
          data = torch.from_numpy(np.nan_to_num(pd.read_csv(path)[features].to_numpy().reshape(T, W, H, -1), nan=self.nan_fill))
          self.listLabels.append(label)
          sample_file='/'.join(str(path).split('/')[-2:])
          sample_file = self.args.ZPI_dir + "/" + sample_file
          postfix = "_zpi_maxDimHoles_"+str(self.args.maxDimHoles[0])+".npz"
          ZPIH0H1 = np.load(sample_file.replace('.csv', postfix))
          ZPI = ZPIH0H1['zpi']
          ZPI = torch.from_numpy(ZPI)
          if trans is not None:
            ZPI = transRot(ZPI)
            data = trans(data)
          self.listZPI.append(ZPI)
          self.listData.append(data)

    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, idx):
        return self.listData[idx], self.listLabels[idx], self.listZPI[idx]
#        data, labels, _ = self.path_list[idx]
#        return data, label, self.listZPI[idx]
#        data = self.prepareData(path)
#        sample_file = path
#        sample_file='/'.join(str(sample_file).split('/')[-4:])
#        sample_file = self.args.ZPI_dir + "/" + sample_file
#        ZPI=[]
#        for i in range(len(self.args.scaleParameter)):
#          postfix = "zpi_"+"scaleParameter_"+str(self.args.scaleParameter[i])+"_maxDimHoles_"+str(self.args.maxDimHoles[i])+"_sizeBorder_"+str(self.args.sizeBorder[i])+".npz"
#          postfix = "zpi_maxDimHoles_"+str(self.args.maxDimHoles[i])+"_sizeBorder_"+str(self.args.sizeBorder[i])+".npz"
#          ZPIH0H1 = np.load(sample_file.replace('dynamic.npy', postfix))
#          ZPI.append(ZPIH0H1['zpi'])
#        ZPI = np.concatenate(ZPI, axis=0)
##        ZPI = np.concatenate(ZPI, axis=0)
#        ZPI = torch.from_numpy(ZPI)

###        (W, H) = (25, 25)
###        (T, F, N) = data.shape
###        data = torch.from_numpy(data)
###        if trans is not None:
###           data = data.reshape(T*F,W,H)
###           data = trans(data)
###           resize_transform = transforms.Resize(size=(W, H))
###           if data.shape[-1] != H:
###             data = resize_transform(data)
###           data = data.reshape(T, F, W*H)        
###           if self.listZPI[idx] is not None:
###              ZPI = self.listZPI[idx](ZPI)
#        return data, labels, self.listZPI[idx]
#        return data, labels, ZPI

def get_dataloaders(args):

  dataFireModule = FireDSDataModuleGCN(args)
  train_dataloader = dataFireModule.train_dataloader()

  val_dataloader = dataFireModule.val_dataloader()

  test_dataloader1 = dataFireModule.test_dataloader1()
  test_dataloader2 = dataFireModule.test_dataloader2()

#  print('Train: x y ->', x_tra.shape, topo_tra.shape, y_tra.shape)
#  print('Val: x, y ->', x_val.shape, topo_val.shape, y_val.shape)
#  print('Test: x, ZPI, y ->', x_test.shape, topo_test.shape, y_test.shape)
   ######################get triple dataloader######################

  return train_dataloader, val_dataloader, test_dataloader1, test_dataloader2
