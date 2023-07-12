#from __future__ import print_function, division

import torch
import numpy as np
import random
from PIL import Image
from torch.utils.data import Dataset
import os
import os.path
from torch.utils.data.sampler import Sampler
import torchvision
from torch.autograd import Variable

def make_dataset(image_list, labels):
    if labels:
      len_ = len(image_list)
      images = [(image_list[i].strip(), labels[i, :]) for i in range(len_)]
    else:
      if len(image_list[0].split()) > 2:
        images = [(val.split()[0], np.array([int(la) for la in val.split()[1:]])) for val in image_list]
      else:
        images = [(val.split()[0], int(val.split()[1])) for val in image_list]
    return images


def rgb_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def l_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('L')

class ImageList(Dataset):
    def __init__(self, image_list, labels=None, transform=None, target_transform=None, mode='RGB'):
        imgs = make_dataset(image_list, labels)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        if mode == 'RGB':
            self.loader = rgb_loader
        elif mode == 'L':
            self.loader = l_loader

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imgs)

class ImageList_idx(Dataset):
    def __init__(self, image_list, labels=None, transform=None, target_transform=None, mode='RGB'):
        imgs = make_dataset(image_list, labels)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        if mode == 'RGB':
            self.loader = rgb_loader
        elif mode == 'L':
            self.loader = l_loader

    def __getitem__(self, index,genererate_data):
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

    def __len__(self):
        return len(self.imgs)

def pair_data(seed,class_num,shot,Y_s,Y_t):
    #change seed so every time wo get group data will different in source domain,but in target domain, data not change
    torch.manual_seed(1 + seed)
    torch.cuda.manual_seed(1 + seed)
    #shuffle order
    classes = torch.unique(Y_t)
    classes=classes[torch.randperm(len(classes))]        
    def s_idxs(c):
        idx=torch.nonzero(Y_s.eq(int(c)))

        return idx[torch.randperm(len(idx))][:shot*2].squeeze()
    def t_idxs(c):
        return torch.nonzero(Y_t.eq(int(c)))[:shot].squeeze()

    source_idxs = list(map(s_idxs, classes))
    target_idxs = list(map(t_idxs, classes))

    source_matrix=torch.stack(source_idxs)
    target_matrix=torch.stack(target_idxs)
    pair_list=[]
    for i in range(class_num):
        for j in range(shot):
            pair_list.append((source_matrix[i][j*2],source_matrix[i][j*2+1]))
    for i in range(class_num):
        for j in range(shot):
            if shot > 1:
                pair_list.append((source_matrix[i][j],target_matrix[i][j]))
            else:
                pair_list.append((source_matrix[i][j],target_matrix[i]))
    for i in range(class_num):
        for j in range(shot):
            pair_list.append((source_matrix[i%class_num][j],source_matrix[(i+1)%class_num][j]))
    for i in range(class_num):
        for j in range(shot):
            if shot > 1:
                pair_list.append((source_matrix[i%class_num][j],target_matrix[(i+1)%class_num][j]))
            else:
                pair_list.append((source_matrix[i%class_num][j],target_matrix[(i+1)%class_num]))   
    return pair_list

def generate_data(generator,target,generate_batch,class_num,g_input_dim):
    label_ = np.ones(generate_batch,dtype=np.int32)*target
    class_onehot = np.zeros((generate_batch,class_num))
    class_onehot[np.arange(generate_batch), label_] = 1
    z_ = np.random.normal(0, 1, (generate_batch, g_input_dim))
    z_[np.arange(generate_batch), :class_num] = class_onehot[np.arange(generate_batch)]
    z_ = (torch.from_numpy(z_).float())
    z_=z_.view(generate_batch,  g_input_dim, 1, 1)
    z_ = Variable(z_.cuda())
    return generator(z_)
class GroupData(Dataset):
    def __init__(self, image_list,generator,generate_batch,class_num,shot,seed=0, labels=None, transform=None, target_transform=None, mode='RGB'):
        imgs = make_dataset(image_list, labels)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))
       
        #group pair
        self.Y_t=[x[1] for x in imgs]
        self.X_s = torch.Tensor(class_num*generate_batch,3,224,224)
        self.Y_s = torch.LongTensor(class_num*generate_batch)

        for target in range(class_num):
            X = generate_data(generator,target,generate_batch,class_num,100+class_num)
            Y = target * torch.ones(generate_batch, dtype = torch.uint8)
            self.X_s[target*generate_batch:(target+1)*generate_batch] = X
            self.Y_s[target*generate_batch:(target+1)*generate_batch] = Y
        self.X_s=self.X_s.detach()
        self.Y_s=self.Y_s.detach()
        self.pair_list=pair_data(seed,class_num,shot,self.Y_s,torch.tensor(self.Y_t))
        self.shot=shot
        self.class_num=class_num
        self.generate_batch=generate_batch
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        if mode == 'RGB':
            self.loader = rgb_loader
        elif mode == 'L':
            self.loader = l_loader
   

   
    def __getitem__(self, index):
        group_num=self.shot*self.class_num
        if index<group_num:
            #group1 gxg
            idx1,idx2=self.pair_list[index]
            img1=self.X_s[idx1]
            img2=self.X_s[idx2]
            label1=int(self.Y_s[idx1])
            label2=int(self.Y_s[idx2])
            gourp_label=0
        elif group_num<=index<2*group_num:
            #group2 gxt
            idx1,idx2=self.pair_list[index]
            img1=self.X_s[idx1]
            label1=int(self.Y_s[idx1])
            path, label2 = self.imgs[idx2]
            img2 = self.loader(path)
            gourp_label=1
            if self.transform is not None:
                img2 = self.transform(img2)
        elif 2*group_num<=index<3*group_num:
            #group3 gxg
            idx1,idx2=self.pair_list[index]
            img1=self.X_s[idx1]
            img2=self.X_s[idx2]
            label1=int(self.Y_s[idx1])
            label2=int(self.Y_s[idx2])
            gourp_label=2
        elif 3*group_num<=index<4*group_num:
            #group4 gxt
             #group2 gxt
            idx1,idx2=self.pair_list[index]
            img1=self.X_s[idx1]
            label1=int(self.Y_s[idx1])
            path, label2 = self.imgs[idx2]
            img2 = self.loader(path)
            gourp_label=3
            if self.transform is not None:
                img2 = self.transform(img2)

      

        return img1, img2,label1,label2,gourp_label

    def __len__(self):
        return len(self.pair_list)
    
    def shuffle(self,seed):
        index = torch.randperm(self.class_num*self.generate_batch)
        self.X_s = self.X_s[index]
        self.Y_s = self.Y_s[index]
        self.pair_list=pair_data(seed,self.class_num,self.shot,self.Y_s,torch.tensor(self.Y_t))

   
    
#collapse-hide
import random
from torch.utils.data.sampler import Sampler
 
class Group24Sampler(Sampler):
    def __init__(self, class_num,shot):
        self.shot=shot
        self.class_num=class_num
        
    def __iter__(self):
        s1=torch.randperm(self.shot*self.class_num)
        group2_sampler=[self.shot*self.class_num+x for x in s1]
        s2=torch.randperm(self.shot*self.class_num)
        group4_sampler=[self.shot*self.class_num*3+x for x in s2]
        sampler=group2_sampler+group4_sampler
        random.shuffle(sampler)
        return iter(sampler)
    
    def __len__(self):
        return self.shot*self.class_num*2
 
 
