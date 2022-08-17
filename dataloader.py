# load all into cpu
# do cropping to patch size
# restrict number of slices (see null slices)
# normalize range
# test code by outputting few patches
# training, testing, val
import torch
import os, glob, sys
import numpy as np
from PIL import Image
from os.path import join as pjoin
from torchvision import transforms
from torch.utils import data
from skimage import io

class CREMI(data.Dataset):
    def __init__(self, listpath, filepaths, is_training=False):
    
        self.listpath = listpath
        self.imgfile = filepaths[0]
        self.gtfile = filepaths[1]

        self.dataCPU = {}
        self.dataCPU['image'] = []
        self.dataCPU['label'] = []

        self.indices = []
        self.crop_size = 128
        self.is_training = is_training

        self.loadCPU()

    def loadCPU(self):
        
        img = io.imread(self.imgfile)
        gt = io.imread(self.gtfile) 

        _, h,w = np.shape(img)
        img_tiff = np.zeros((h,w))
        gt_tiff = np.zeros((h,w))
        img_tiff = np.array(img)
        gt_tiff = 1 - np.array(gt)/255

        img = torch.tensor(img_tiff)
        gt = torch.tensor(gt_tiff)   

        with open(self.listpath, 'r') as f:
            mylist = f.readlines()
        mylist = [x.rstrip('\n') for x in mylist]

        for i, entry in enumerate(mylist): 

            meanval = torch.Tensor.float(img[int(entry)]).mean()
            stdval = torch.Tensor.float(img[int(entry)]).std()
            
            self.indices.append((i))

            #cpu store
            self.dataCPU['image'].append((img[int(entry)] - meanval) / stdval)
            self.dataCPU['label'].append(gt[int(entry)])


    def __len__(self): # total number of 2D slices
        return len(self.indices)

    def __getitem__(self, index): # return CHW torch tensor
        index = self.indices[index]

        torch_img = self.dataCPU['image'][index] #HW
        torch_gt = self.dataCPU['label'][index] #HW

        if self.is_training is True:
            # crop to patchsize. compute top-left corner first
            H, W = torch_img.shape
            corner_h = np.random.randint(low=0, high=H-self.crop_size)
            corner_w = np.random.randint(low=0, high=W-self.crop_size)
            torch_img = torch_img[corner_h:corner_h+self.crop_size, corner_w:corner_w+self.crop_size]
            torch_gt = torch_gt[corner_h:corner_h+self.crop_size, corner_w:corner_w+self.crop_size]
        # else:
        #     torch_img = torch_img[0:1248, 0:1248]
        #     torch_gt = torch_gt[0:1248, 0:1248]

        torch_img = torch.unsqueeze(torch_img,dim=0).repeat(1,1,1)
        torch_gt = torch.unsqueeze(torch_gt,dim=0)
        return torch_img, torch_gt 

class ISBI2013(data.Dataset):
    def __init__(self, listpath, filepaths, is_training=False):
    
        self.listpath = listpath
        self.imgfile = filepaths[0]
        self.gtfile = filepaths[1]

        self.dataCPU = {}
        self.dataCPU['image'] = []
        self.dataCPU['label'] = []

        self.indices = []
        self.crop_size = 128
        self.is_training = is_training

        self.loadCPU()

    def loadCPU(self):
        
        img = io.imread(self.imgfile)
        gt = io.imread(self.gtfile) 

        _, h,w = np.shape(img)
        img_tiff = np.zeros((h,w))
        gt_tiff = np.zeros((h,w))
        img_tiff = np.array(img)
        gt_tiff = 1 - np.array(gt)/255

        img = torch.tensor(img_tiff)
        gt = torch.tensor(gt_tiff)   

        with open(self.listpath, 'r') as f:
            mylist = f.readlines()
        mylist = [x.rstrip('\n') for x in mylist]

        for i, entry in enumerate(mylist): 

            meanval = torch.Tensor.float(img[int(entry)]).mean()
            stdval = torch.Tensor.float(img[int(entry)]).std()

            self.indices.append((i))

            #cpu store
            self.dataCPU['image'].append((img[int(entry)] - meanval) / stdval)
            self.dataCPU['label'].append(gt[int(entry)])


    def __len__(self): # total number of 2D slices
        return len(self.indices)

    def __getitem__(self, index): # return CHW torch tensor
        index = self.indices[index]

        torch_img = self.dataCPU['image'][index] #HW
        torch_gt = self.dataCPU['label'][index] #HW

        if self.is_training is True:
            # crop to patchsize. compute top-left corner first
            H, W = torch_img.shape
            corner_h = np.random.randint(low=0, high=H-self.crop_size)
            corner_w = np.random.randint(low=0, high=W-self.crop_size)
            torch_img = torch_img[corner_h:corner_h+self.crop_size, corner_w:corner_w+self.crop_size]
            torch_gt = torch_gt[corner_h:corner_h+self.crop_size, corner_w:corner_w+self.crop_size]

        torch_img = torch.unsqueeze(torch_img,dim=0).repeat(1,1,1)
        torch_gt = torch.unsqueeze(torch_gt,dim=0)
        return torch_img, torch_gt 

class DRIVE(data.Dataset):
    def __init__(self, listpath, folderpaths, is_training=False):

        self.listpath = listpath
        self.imgfolder = folderpaths[0]
        self.gtfolder = folderpaths[1]

        self.dataCPU = {}
        self.dataCPU['image'] = []
        self.dataCPU['label'] = []

        self.indices = [] 
        self.to_tensor = transforms.ToTensor()
        self.crop_size = 128
        self.is_training = is_training

        self.loadCPU()

    def loadCPU(self):
        with open(self.listpath, 'r') as f:
            mylist = f.readlines()
        mylist = [x.rstrip('\n') for x in mylist]

        for i, entry in enumerate(mylist):

            components = entry.split('.')
            filename = components[0]

            im_path = pjoin(self.imgfolder, filename) + '.tif'
            gt_path = pjoin(self.gtfolder, filename) + '_manual1.gif'
            img = Image.open(im_path)
            gt = Image.open(gt_path)

            img = self.to_tensor(img)
            gt = self.to_tensor(gt)

            #normalize within a channel
            for j in range(img.shape[0]):
                meanval = img[j].mean()
                stdval = img[j].std()
                img[j] = (img[j] - meanval) / stdval

            self.indices.append((i))

            #cpu store
            self.dataCPU['image'].append(img)
            self.dataCPU['label'].append(gt)


    def __len__(self): # total number of 2D slices
        return len(self.indices)

    def __getitem__(self, index): # return CHW torch tensor
        index = self.indices[index]

        torch_img = self.dataCPU['image'][index] #HW
        torch_gt = self.dataCPU['label'][index] #HW

        if self.is_training is True:
            # crop to patchsize. compute top-left corner first
            C, H, W = torch_img.shape
            corner_h = np.random.randint(low=0, high=H-self.crop_size)
            corner_w = np.random.randint(low=0, high=W-self.crop_size)

            torch_img = torch_img[:, corner_h:corner_h+self.crop_size, corner_w:corner_w+self.crop_size]
            torch_gt = torch_gt[:, corner_h:corner_h+self.crop_size, corner_w:corner_w+self.crop_size]

        return torch_img, torch_gt 

if __name__ == "__main__":
    flag = "training"
    
    dst = CREMI('data-lists/CREMI/validation-list.csv', ['data/CREMI/train-volume.tif', 'data/CREMI/train-labels.tif'])
    # dst = ISBI2013('data-lists/ISBI2013/train-list.csv', ['data/ISBI2013/train-volume.tif', 'data/ISBI2013/train-labels.tif'], is_training= True)

    # dst = DRIVE('data-lists/DRIVE/train-list.csv', ['data/DRIVE/images', 'data/DRIVE/1st_manual'], is_training= True)
 
    validationloader = data.DataLoader(dst, shuffle=False, batch_size=1, num_workers=1)

    ## dataloader check
    # import pdb; pdb.set_trace()
    batch = next(iter(validationloader))
    input, target = batch
    # import pdb; pdb.set_trace()
