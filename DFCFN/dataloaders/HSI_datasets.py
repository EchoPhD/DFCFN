import collections
import math
import os
import random
from socket import MsgFlag

import cv2
import numpy as np
import torch
from torch.utils import data

import scipy
import scipy.ndimage
import scipy.io

import h5py
# Pavia dataset (Dataset can be downloaded from the following link)
# http://www.ehu.eus/ccwintco/index.php?title=Hyperspectral_Remote_Sensing_Scenes

class pavia_dataset(data.Dataset):
    def __init__(
        self, config, is_train=True, is_dhp=False, want_DHP_MS_HR=False
    ):
        self.split  = "train" if is_train else "val"        #Define train and validation splits
        self.config = config                                #Configuration file
        self.want_DHP_MS_HR = want_DHP_MS_HR                #This ask: DO we need DIP up-sampled output as dataloader output?
        self.is_dhp         = is_dhp                        #This checks: "Is this DIP training?"
        self.dir = self.config["pavia_dataset"]["data_dir"] #Path to Pavia Center dataset 
        
        if self.split == "val":
            self.file_list = os.path.join(self.dir, f"{self.split}" + ".txt")
        elif self.split == "train":
            self.file_list = os.path.join(self.dir, f"{self.split}" + ".txt")
            if is_dhp:
                self.file_list = os.path.join(self.dir, f"{self.split}_dhp" + ".txt")
        
        self.images = [line.rstrip("\n") for line in open(self.file_list)] #Read image name corresponds to train/val/test set
    
        self.augmentation = self.config["pavia_dataset"]["augmentation"]   #Augmentation needed or not? 

        self.LR_crop_size = (self.config["pavia_dataset"]["LR_size"], self.config["pavia_dataset"]["LR_size"])  #Size of the the LR-HSI

        self.HR_crop_size = [self.config["pavia_dataset"]["HR_size"], self.config["pavia_dataset"]["HR_size"]]  #Size of the HR-HSI

        cv2.setNumThreads(0)    # to avoid Deadloack  between CV Threads and Pytorch Threads caused in resizing

        self.files = collections.defaultdict(list)
        for f in self.images:
            self.img_root = self.dir+f+"/"
            self.files[self.split].append(
                {
                    "imgs": self.img_root + f + ".mat",
                }
            )

    def __len__(self):
        return len(self.files[self.split])
    
    #def _augmentaion(self, MS_image, PAN_image, reference):
    def _augmentaion(self, MS_image, PAN_image, reference,MS_image_up):     ### \\- 2023- 0315 rewrite -\\
        N_augs = 4
        aug_idx = torch.randint(0, N_augs, (1,))
        if aug_idx==0:
            #Horizontal Flip
            MS_image    = torch.flip(MS_image, [1]) 
            MS_image_up    = torch.flip(MS_image, [1])  ### \\- 2023-0315 add -\\
            PAN_image   = torch.flip(PAN_image, [0])
            reference   = torch.flip(reference, [1])
        elif aug_idx==1:
            #Vertical Flip
            MS_image    = torch.flip(MS_image, [2])
            MS_image_up    = torch.flip(MS_image, [2])  ### \\- 2023-0315 add -\\
            PAN_image   = torch.flip(PAN_image, [1])
            reference   = torch.flip(reference, [2])
        elif aug_idx==2:
            #Horizontal flip
            MS_image    = torch.flip(MS_image, [1]) 
            MS_image_up    = torch.flip(MS_image, [1])  ### \\- 2023-0315 add -\\
            PAN_image   = torch.flip(PAN_image, [0])
            reference   = torch.flip(reference, [1])
            #Vertical Flip
            MS_image    = torch.flip(MS_image, [2])
            MS_image_up    = torch.flip(MS_image, [2])  ### \\- 2023-0315 add -\\
            PAN_image   = torch.flip(PAN_image, [1])
            reference   = torch.flip(reference, [2])

        #return MS_image, PAN_image, reference
        return MS_image, PAN_image, reference, MS_image_up  ### \\- 2023-0315 rewrite -\\ 

    def getHSIdata(self, index):
        image_dict = self.files[self.split][index]
       
        # read each image in list
        #mat         = scipy.io.loadmat(image_dict["imgs"])
        mat	= h5py.File(image_dict["imgs"], 'r')
        reference   = mat["ref"]
        PAN_image   = mat["pan"]

        if self.want_DHP_MS_HR:
            opt_lambda  = self.config["pavia_dataset"]["optimal_lambda"]
            #mat_dhp     = scipy.io.loadmat(image_dict["imgs"][:-4]+"_dhp_"+"{0:0=1d}".format(int(10*opt_lambda))+ ".mat")
            # Taking DIP up-sampled image as inputs
            #MS_image = mat_dhp["dhp"]
        else:
            ''' \\\\\\ - 2023-0315 rewrite- \\\\\\ '''
            opt_lambda  = self.config["pavia_dataset"]["optimal_lambda"]
            mat_dhp     = scipy.io.loadmat(image_dict["imgs"][:-4]+"_dhp_"+"{0:0=1d}".format(int(10*opt_lambda))+ ".mat")
            #Taking DIP up-sampled image as inputs
            MS_image_up = mat_dhp["dhp"]
            MS_image = mat["y"]
            #print("HSI_datasets.py --> pavia_dataset: dhp data is loaded!!! ")	### \\ - 2023-0315 add - \\
            ''' \\\\\\ -  - END -  -  \\\\\\ '''
            
        # COnvert inputs into torch tensors
        #MS_image    = torch.from_numpy((np.array(MS_image)/1.0).transpose(2, 0, 1))
        #MS_image_up   = torch.from_numpy((np.array(MS_image_up)/1.0).transpose(2, 0, 1))    ### \\- 2023-0315 add -\\
        #PAN_image   = torch.from_numpy(np.array(PAN_image)/1.0)
        #reference   = torch.from_numpy((np.array(reference)/1.0).transpose(2, 0, 1))
        
        #MS_image    = torch.from_numpy((np.array(MS_image)/1.0).transpose(2, 0, 1)) ###用DHP的话，打开这一行
        MS_image    = torch.from_numpy((np.array(MS_image)/1.0).transpose(0, 2, 1)) ###不用DHP的话，打开这一行
        MS_image_up   = torch.from_numpy((np.array(MS_image_up)/1.0).transpose(2, 0, 1))    ### \\- 2023-0315 add -\\
        PAN_image   = torch.from_numpy((np.array(PAN_image)/1.0).transpose(1,0))
        reference   = torch.from_numpy((np.array(reference)/1.0).transpose(0, 2, 1))
        
        #print("MS_image.shape:", MS_image.shape)
        #print("MS_image_up.shape:", MS_image_up.shape)
        #print("PAN_image.shape:", PAN_image.shape)
        #print("reference.shape:", reference.shape)
              
        # Max Normalization
        MS_image    = MS_image/self.config["pavia_dataset"]["max_value"]
        MS_image_up    = MS_image_up/self.config["pavia_dataset"]["max_value"]   ### \\- 2023-0315 add -\\
        PAN_image   = PAN_image/self.config["pavia_dataset"]["max_value"]
        reference   = reference/self.config["pavia_dataset"]["max_value"]           

        #If split = "train" and augment = "true" do augmentation
        if self.split == "train" and self.augmentation:
            #MS_image, PAN_image, reference = self._augmentaion(MS_image, PAN_image, reference)
            MS_image, PAN_image, reference, MS_image_up = self._augmentaion(MS_image, PAN_image, reference, MS_image_up)    ### \\- 2023-0315 rewrite -\\
            #print("HSI_datasets.py --> pavia_dataset: augmentation has done!!! ")	### \\ -2023-0315 add - \\
        if self.split == "train" and index == len(self.files[self.split]) - 1:
            np.random.shuffle(self.files[self.split])

        #return image_dict, MS_image, PAN_image, reference
        return image_dict, MS_image, PAN_image, reference, MS_image_up ### \\- 2023-0315 rewrite -\\

    
    def __getitem__(self, index):

        #image_dict, MS_image, PAN_image, reference = self.getHSIdata(index)
        image_dict, MS_image, PAN_image, reference, MS_image_up = self.getHSIdata(index)    ### \\- 2023-0315 rewrite -\\
        #return image_dict, MS_image, PAN_image, reference
        return image_dict, MS_image, PAN_image, reference, MS_image_up ### \\- 2023-0315 rewrite -\\


class paviaU_dataset(data.Dataset):
    def __init__(
        self, config, is_train=True, is_dhp=False, want_DHP_MS_HR=False
    ):
        self.split  = "train" if is_train else "val"        #Define train and validation splits
        self.config = config                                #Configuration file
        self.want_DHP_MS_HR = want_DHP_MS_HR                #This ask: DO we need DIP up-sampled output as dataloader output?
        self.is_dhp         = is_dhp                        #This checks: "Is this DIP training?"
        self.dir = self.config["paviaU_dataset"]["data_dir"] #Path to Pavia Center dataset 
        
        if self.split == "val":
            self.file_list = os.path.join(self.dir, f"{self.split}" + ".txt")
        elif self.split == "train":
            self.file_list = os.path.join(self.dir, f"{self.split}" + ".txt")
            if is_dhp:
                self.file_list = os.path.join(self.dir, f"{self.split}_dhp" + ".txt")
        
        self.images = [line.rstrip("\n") for line in open(self.file_list)] #Read image name corresponds to train/val/test set
    
        self.augmentation = self.config["paviaU_dataset"]["augmentation"]   #Augmentation needed or not? 

        self.LR_crop_size = (self.config["paviaU_dataset"]["LR_size"], self.config["paviaU_dataset"]["LR_size"])  #Size of the the LR-HSI

        self.HR_crop_size = [self.config["paviaU_dataset"]["HR_size"], self.config["paviaU_dataset"]["HR_size"]]  #Size of the HR-HSI

        cv2.setNumThreads(0)    # to avoid Deadloack  between CV Threads and Pytorch Threads caused in resizing

        self.files = collections.defaultdict(list)
        for f in self.images:
            self.img_root = self.dir+f+"/"
            self.files[self.split].append(
                {
                    "imgs": self.img_root + f + ".mat",
                }
            )

    def __len__(self):
        return len(self.files[self.split])
    
    #def _augmentaion(self, MS_image, PAN_image, reference):
    def _augmentaion(self, MS_image, PAN_image, reference,MS_image_up):     ### \\- 2023- 0315 rewrite -\\
        N_augs = 4
        aug_idx = torch.randint(0, N_augs, (1,))
        if aug_idx==0:
            #Horizontal Flip
            MS_image    = torch.flip(MS_image, [1]) 
            MS_image_up    = torch.flip(MS_image, [1])  ### \\- 2023-0315 add -\\
            PAN_image   = torch.flip(PAN_image, [0])
            reference   = torch.flip(reference, [1])
        elif aug_idx==1:
            #Vertical Flip
            MS_image    = torch.flip(MS_image, [2])
            MS_image_up    = torch.flip(MS_image, [2])  ### \\- 2023-0315 add -\\
            PAN_image   = torch.flip(PAN_image, [1])
            reference   = torch.flip(reference, [2])
        elif aug_idx==2:
            #Horizontal flip
            MS_image    = torch.flip(MS_image, [1]) 
            MS_image_up    = torch.flip(MS_image, [1])  ### \\- 2023-0315 add -\\
            PAN_image   = torch.flip(PAN_image, [0])
            reference   = torch.flip(reference, [1])
            #Vertical Flip
            MS_image    = torch.flip(MS_image, [2])
            MS_image_up    = torch.flip(MS_image, [2])  ### \\- 2023-0315 add -\\
            PAN_image   = torch.flip(PAN_image, [1])
            reference   = torch.flip(reference, [2])

        #return MS_image, PAN_image, reference
        return MS_image, PAN_image, reference, MS_image_up  ### \\- 2023-0315 rewrite -\\ 

    def getHSIdata(self, index):
        image_dict = self.files[self.split][index]
       
        # read each image in list
        #mat         = scipy.io.loadmat(image_dict["imgs"])
        mat	= h5py.File(image_dict["imgs"], 'r')
        reference   = mat["ref"]
        PAN_image   = mat["pan"]

        if self.want_DHP_MS_HR:
            opt_lambda  = self.config["paviaU_dataset"]["optimal_lambda"]
            #mat_dhp     = scipy.io.loadmat(image_dict["imgs"][:-4]+"_dhp_"+"{0:0=1d}".format(int(10*opt_lambda))+ ".mat")
            # Taking DIP up-sampled image as inputs
            #MS_image = mat_dhp["dhp"]
        else:
            ''' \\\\\\ - 2023-0315 rewrite- \\\\\\ '''
            opt_lambda  = self.config["paviaU_dataset"]["optimal_lambda"]
            mat_dhp     = scipy.io.loadmat(image_dict["imgs"][:-4]+"_dhp_"+"{0:0=1d}".format(int(10*opt_lambda))+ ".mat")
            #Taking DIP up-sampled image as inputs
            MS_image_up = mat_dhp["dhp"]
            MS_image = mat["y"]
            #print("HSI_datasets.py --> pavia_dataset: dhp data is loaded!!! ")	### \\ - 2023-0315 add - \\
            ''' \\\\\\ -  - END -  -  \\\\\\ '''
            
        # COnvert inputs into torch tensors
        #MS_image    = torch.from_numpy((np.array(MS_image)/1.0).transpose(2, 0, 1))
        #MS_image_up   = torch.from_numpy((np.array(MS_image_up)/1.0).transpose(2, 0, 1))    ### \\- 2023-0315 add -\\
        #PAN_image   = torch.from_numpy(np.array(PAN_image)/1.0)
        #reference   = torch.from_numpy((np.array(reference)/1.0).transpose(2, 0, 1))
        
        #MS_image    = torch.from_numpy((np.array(MS_image)/1.0).transpose(2, 0, 1)) 
        MS_image    = torch.from_numpy((np.array(MS_image)/1.0).transpose(0, 2, 1)) 
        MS_image_up   = torch.from_numpy((np.array(MS_image_up)/1.0).transpose(2, 0, 1))    ### \\- 2023-0315 add -\\
        PAN_image   = torch.from_numpy((np.array(PAN_image)/1.0).transpose(1,0))
        reference   = torch.from_numpy((np.array(reference)/1.0).transpose(0, 2, 1))      
        
        
        # Max Normalization
        MS_image    = MS_image/self.config["paviaU_dataset"]["max_value"]
        MS_image_up    = MS_image_up/self.config["paviaU_dataset"]["max_value"]   ### \\- 2023-0315 add -\\
        PAN_image   = PAN_image/self.config["paviaU_dataset"]["max_value"]
        reference   = reference/self.config["paviaU_dataset"]["max_value"]           

        #If split = "train" and augment = "true" do augmentation
        if self.split == "train" and self.augmentation:
            #MS_image, PAN_image, reference = self._augmentaion(MS_image, PAN_image, reference)
            MS_image, PAN_image, reference, MS_image_up = self._augmentaion(MS_image, PAN_image, reference, MS_image_up)    ### \\- 2023-0315 rewrite -\\
            #print("HSI_datasets.py --> pavia_dataset: augmentation has done!!! ")	### \\ -2023-0315 add - \\
        if self.split == "train" and index == len(self.files[self.split]) - 1:
            np.random.shuffle(self.files[self.split])

        #return image_dict, MS_image, PAN_image, reference
        return image_dict, MS_image, PAN_image, reference, MS_image_up ### \\- 2023-0315 rewrite -\\

    
    def __getitem__(self, index):

        #image_dict, MS_image, PAN_image, reference = self.getHSIdata(index)
        image_dict, MS_image, PAN_image, reference, MS_image_up = self.getHSIdata(index)    ### \\- 2023-0315 rewrite -\\
        #return image_dict, MS_image, PAN_image, reference
        return image_dict, MS_image, PAN_image, reference, MS_image_up ### \\- 2023-0315 rewrite -\\

class chikusei_dataset(data.Dataset):
    def __init__(
        self, config, is_train=True, is_dhp=False, want_DHP_MS_HR=False
    ):
        # Settings
        self.split          = "train" if is_train else "val"
        self.config         = config
        self.want_DHP_MS_HR = want_DHP_MS_HR
        self.is_dhp         = is_dhp

        # Paths
        self.dir            = self.config["chikusei_dataset"]["data_dir"]

        # Read train/val image indexes from the text file (train.txt, val.txt, train_dhp.txt)
        if self.split == "val":
            self.file_list = os.path.join(self.dir, f"{self.split}" + ".txt")
        elif self.split == "train":
            self.file_list = os.path.join(self.dir, f"{self.split}" + ".txt")
            if is_dhp:
                self.file_list = os.path.join(self.dir, f"{self.split}_dhp" + ".txt")

        # list of all images
        self.images = [line.rstrip("\n") for line in open(self.file_list)]

        # augmentations
        self.augmentation = self.config["chikusei_dataset"]["augmentation"]

        self.LR_crop_size = (self.config["chikusei_dataset"]["LR_size"], self.config["chikusei_dataset"]["LR_size"])

        self.HR_crop_size = [
            self.config["chikusei_dataset"]["HR_size"],
            self.config["chikusei_dataset"]["HR_size"],
        ]

        # to avoid Deadloack  between CV Threads and Pytorch Threads caused in resizing
        cv2.setNumThreads(0)

        self.files = collections.defaultdict(list)
        for f in self.images:
            self.img_root = self.dir+f+"/"
            self.files[self.split].append(
                {
                    "imgs": self.img_root + f + ".mat",
                }
            )

    def __len__(self):
        return len(self.files[self.split])

    def getHSIdata(self, index):
        image_dict = self.files[self.split][index]
       
        # read each image in list
        #mat         = scipy.io.loadmat(image_dict["imgs"])
        mat	= h5py.File(image_dict["imgs"], 'r')
        reference   = mat["ref"]
        PAN_image   = mat["pan"]

        if self.want_DHP_MS_HR:
            opt_lambda  = self.config["chikusei_dataset"]["optimal_lambda"]
            #mat_dhp     = scipy.io.loadmat(image_dict["imgs"][:-4]+"_dhp_"+"{0:0=1d}".format(int(10*opt_lambda))+ ".mat")
            mat_dhp	= h5py.File(image_dict["imgs"][:-4]+"_dhp_"+"{0:0=1d}".format(int(10*opt_lambda))+ ".mat", 'r') 
            
            # Taking DIP up-sampled image as inputs
            MS_image = mat_dhp["dhp"]
        else:
            MS_image = mat["y"]
            
        if self.want_DHP_MS_HR:
            opt_lambda  = self.config["chikusei_dataset"]["optimal_lambda"]
            #mat_dhp     = scipy.io.loadmat(image_dict["imgs"][:-4]+"_dhp_"+"{0:0=1d}".format(int(10*opt_lambda))+ ".mat")
            # Taking DIP up-sampled image as inputs
            #MS_image = mat_dhp["dhp"]
        else:
            ''' \\\\\\ - 2023-0315 rewrite- \\\\\\ '''
            opt_lambda  = self.config["chikusei_dataset"]["optimal_lambda"]
            mat_dhp     = scipy.io.loadmat(image_dict["imgs"][:-4]+"_dhp_"+"{0:0=1d}".format(int(10*opt_lambda))+ ".mat")
            #Taking DIP up-sampled image as inputs
            MS_image_up = mat_dhp["dhp"]
            MS_image = mat["y"]
            #print("HSI_datasets.py --> chikusei_dataset: dhp data is loaded!!! ")	### \\ - 2023-0315 add - \\
            ''' \\\\\\ -  - END -  -  \\\\\\ '''
            
        # COnvert inputs into torch tensors
        #MS_image    = torch.from_numpy((np.array(MS_image)/1.0).transpose(2, 0, 1))
        #MS_image_up   = torch.from_numpy((np.array(MS_image_up)/1.0).transpose(2, 0, 1))    ### \\- 2023-0315 add -\\
        #PAN_image   = torch.from_numpy(np.array(PAN_image)/1.0)
        #reference   = torch.from_numpy((np.array(reference)/1.0).transpose(2, 0, 1))
        
        #MS_image    = torch.from_numpy((np.array(MS_image)/1.0).transpose(2, 0, 1)) ###用DHP的话，打开这一行
        MS_image    = torch.from_numpy((np.array(MS_image)/1.0).transpose(0, 2, 1)) ###不用DHP的话，打开这一行
        MS_image_up   = torch.from_numpy((np.array(MS_image_up)/1.0).transpose(2, 0, 1))    ### \\- 2023-0315 add -\\
        PAN_image   = torch.from_numpy((np.array(PAN_image)/1.0).transpose(1,0))
        reference   = torch.from_numpy((np.array(reference)/1.0).transpose(0, 2, 1)) 
        
        # Max Normalization
        MS_image    = MS_image/self.config["chikusei_dataset"]["max_value"]
        MS_image_up    = MS_image_up/self.config["chikusei_dataset"]["max_value"]   ### \\- 2023-0315 add -\\
        PAN_image   = PAN_image/self.config["chikusei_dataset"]["max_value"]
        reference   = reference/self.config["chikusei_dataset"]["max_value"]           

        if self.split == "train" and index == len(self.files[self.split]) - 1:
            np.random.shuffle(self.files[self.split])         

        #return image_dict, MS_image, PAN_image, reference
        return image_dict, MS_image, PAN_image, reference, MS_image_up ### \\- 2023-0315 rewrite -\\

    def __getitem__(self, index):

        #image_dict, MS_image, PAN_image, reference = self.getHSIdata(index)
        image_dict, MS_image, PAN_image, reference, MS_image_up = self.getHSIdata(index)    ### \\- 2023-0315 rewrite -\\
        #return image_dict, MS_image, PAN_image, reference
        return image_dict, MS_image, PAN_image, reference, MS_image_up ### \\- 2023-0315 rewrite -\\


###  Loss Angleses Dataset ### 
class la_dataset(data.Dataset):
    def __init__(
        self, config, is_train=True, is_dhp=False, want_DHP_MS_HR=False
    ):
        # Settings
        self.split          = "train" if is_train else "val"
        self.config         = config
        self.want_DHP_MS_HR = want_DHP_MS_HR
        self.is_dhp         = is_dhp

        # Paths
        self.dir            = self.config["la_dataset"]["data_dir"]

        # Read train/val image indexes from the text file (train.txt, val.txt, train_dhp.txt)
        if self.split == "val":
            self.file_list = os.path.join(self.dir, f"{self.split}" + ".txt")
        elif self.split == "train":
            self.file_list = os.path.join(self.dir, f"{self.split}" + ".txt")
            if is_dhp:
                self.file_list = os.path.join(self.dir, f"{self.split}_dhp" + ".txt")

        # list of all images
        self.images = [line.rstrip("\n") for line in open(self.file_list)]

        # augmentations
        self.augmentation = self.config["la_dataset"]["augmentation"]

        self.LR_crop_size = (self.config["la_dataset"]["LR_size"], self.config["la_dataset"]["LR_size"])

        self.HR_crop_size = [
            self.config["la_dataset"]["HR_size"],
            self.config["la_dataset"]["HR_size"],
        ]

        # to avoid Deadloack  between CV Threads and Pytorch Threads caused in resizing
        cv2.setNumThreads(0)

        self.files = collections.defaultdict(list)
        for f in self.images:
            self.img_root = self.dir+f+"/"
            self.files[self.split].append(
                {
                    "imgs": self.img_root + f + ".mat",
                }
            )

    def __len__(self):
        return len(self.files[self.split])

    def getHSIdata(self, index):
        image_dict = self.files[self.split][index]
       
        # read each image in list
        mat         = scipy.io.loadmat(image_dict["imgs"])
        reference   = mat["ref"]
        PAN_image   = mat["pan"]

        if self.want_DHP_MS_HR:
            opt_lambda  = self.config["la_dataset"]["optimal_lambda"]
            mat_dhp     = scipy.io.loadmat(image_dict["imgs"][:-4]+"_dhp_"+"{0:0=1d}".format(int(10*opt_lambda))+ ".mat")
            
            # Taking DIP up-sampled image as inputs
            MS_image = mat_dhp["dhp"]
        else:
            MS_image = mat["y"]
            
        # COnvert inputs into torch tensors
        MS_image    = torch.from_numpy((np.array(MS_image)/1.0).transpose(2, 0, 1))
        PAN_image   = torch.from_numpy(np.array(PAN_image)/1.0)
        reference   = torch.from_numpy((np.array(reference)/1.0).transpose(2, 0, 1))
        
        # Max Normalization
        MS_image    = MS_image/self.config["la_dataset"]["max_value"]
        PAN_image   = PAN_image/self.config["la_dataset"]["max_value"]
        reference   = reference/self.config["la_dataset"]["max_value"]           

        if self.split == "train" and index == len(self.files[self.split]) - 1:
            np.random.shuffle(self.files[self.split])

        return image_dict, MS_image, PAN_image, reference

    
    def __getitem__(self, index):

        image_dict, MS_image, PAN_image, reference = self.getHSIdata(index)

        return image_dict, MS_image, PAN_image, reference

###  Botswana (x4) Dataset  ###
class botswana4_dataset(data.Dataset):
    def __init__(
        self, config, is_train=True, is_dhp=False, want_DHP_MS_HR=False
    ):
        self.split  = "train" if is_train else "val"        #Define train and validation splits
        self.config = config                                #Configuration file
        self.want_DHP_MS_HR = want_DHP_MS_HR                #This ask: DO we need DIP up-sampled output as dataloader output?
        self.is_dhp         = is_dhp                        #This checks: "Is this DIP training?"
        self.dir = self.config["botswana4_dataset"]["data_dir"] #Path to Pavia Center dataset 
        
        if self.split == "val":
            self.file_list = os.path.join(self.dir, f"{self.split}" + ".txt")
        elif self.split == "train":
            self.file_list = os.path.join(self.dir, f"{self.split}" + ".txt")
            if is_dhp:
                self.file_list = os.path.join(self.dir, f"{self.split}_dhp" + ".txt")
        
        self.images = [line.rstrip("\n") for line in open(self.file_list)] #Read image name corresponds to train/val/test set
    
        self.augmentation = self.config["botswana4_dataset"]["augmentation"]   #Augmentation needed or not? 

        self.LR_crop_size = (self.config["botswana4_dataset"]["LR_size"], self.config["botswana4_dataset"]["LR_size"])  #Size of the the LR-HSI

        self.HR_crop_size = [self.config["botswana4_dataset"]["HR_size"], self.config["botswana4_dataset"]["HR_size"]]  #Size of the HR-HSI

        cv2.setNumThreads(0)    # to avoid Deadloack  between CV Threads and Pytorch Threads caused in resizing

        self.files = collections.defaultdict(list)
        for f in self.images:
            self.img_root = self.dir+f+"/"
            self.files[self.split].append(
                {
                    "imgs": self.img_root + f + ".mat",
                }
            )

    def __len__(self):
        return len(self.files[self.split])
    
    def _augmentaion(self, MS_image, PAN_image, reference):
        N_augs = 4
        aug_idx = torch.randint(0, N_augs, (1,))
        if aug_idx==0:
            #Horizontal Flip
            MS_image    = torch.flip(MS_image, [1]) 
            PAN_image   = torch.flip(PAN_image, [0])
            reference   = torch.flip(reference, [1])
        elif aug_idx==1:
            #Vertical Flip
            MS_image    = torch.flip(MS_image, [2])
            PAN_image   = torch.flip(PAN_image, [1])
            reference   = torch.flip(reference, [2])
        elif aug_idx==2:
            #Horizontal flip
            MS_image    = torch.flip(MS_image, [1]) 
            PAN_image   = torch.flip(PAN_image, [0])
            reference   = torch.flip(reference, [1])
            #Vertical Flip
            MS_image    = torch.flip(MS_image, [2])
            PAN_image   = torch.flip(PAN_image, [1])
            reference   = torch.flip(reference, [2])

        return MS_image, PAN_image, reference

    def getHSIdata(self, index):
        image_dict = self.files[self.split][index]
       
        # read each image in list
        #mat         = scipy.io.loadmat(image_dict["imgs"])
        mat	= h5py.File(image_dict["imgs"], 'r')
        reference   = mat["ref"]
        PAN_image   = mat["pan"]

        if self.want_DHP_MS_HR:
            opt_lambda  = self.config["botswana4_dataset"]["optimal_lambda"]
            #mat_dhp     = scipy.io.loadmat(image_dict["imgs"][:-4]+"_dhp_"+"{0:0=1d}".format(int(10*opt_lambda))+ ".mat")
            # Taking DIP up-sampled image as inputs
            #MS_image = mat_dhp["dhp"]
        else:
            ''' \\\\\\ - 2023-0315 rewrite- \\\\\\ '''
            opt_lambda  = self.config["botswana4_dataset"]["optimal_lambda"]
            mat_dhp     = scipy.io.loadmat(image_dict["imgs"][:-4]+"_dhp_"+"{0:0=1d}".format(int(10*opt_lambda))+ ".mat")
            #Taking DIP up-sampled image as inputs
            MS_image_up = mat_dhp["dhp"]
            MS_image = mat["y"]
            #print("HSI_datasets.py --> pavia_dataset: dhp data is loaded!!! ")	### \\ - 2023-0315 add - \\
            ''' \\\\\\ -  - END -  -  \\\\\\ '''


        # COnvert inputs into torch tensors
        #MS_image    = torch.from_numpy((np.array(MS_image)/1.0).transpose(2, 0, 1))
        #MS_image_up   = torch.from_numpy((np.array(MS_image_up)/1.0).transpose(2, 0, 1))    ### \\- 2023-0315 add -\\
        #PAN_image   = torch.from_numpy(np.array(PAN_image)/1.0)
        #reference   = torch.from_numpy((np.array(reference)/1.0).transpose(2, 0, 1))
        
        #MS_image    = torch.from_numpy((np.array(MS_image)/1.0).transpose(2, 0, 1)) ###用DHP的话，打开这一行
        MS_image    = torch.from_numpy((np.array(MS_image)/1.0).transpose(0, 2, 1)) ###不用DHP的话，打开这一行
        MS_image_up   = torch.from_numpy((np.array(MS_image_up)/1.0).transpose(2, 0, 1))    ### \\- 2023-0315 add -\\
        PAN_image   = torch.from_numpy((np.array(PAN_image)/1.0).transpose(1,0))
        reference   = torch.from_numpy((np.array(reference)/1.0).transpose(0, 2, 1)) 
        
        # Max Normalization
        MS_image    = MS_image/self.config["botswana4_dataset"]["max_value"]
        MS_image_up    = MS_image_up/self.config["botswana4_dataset"]["max_value"]   ### \\- 2023-0315 add -\\
        PAN_image   = PAN_image/self.config["botswana4_dataset"]["max_value"]
        reference   = reference/self.config["botswana4_dataset"]["max_value"]           

        #If split = "train" and augment = "true" do augmentation
        if self.split == "train" and self.augmentation:
            #MS_image, PAN_image, reference = self._augmentaion(MS_image, PAN_image, reference)
            MS_image, PAN_image, reference, MS_image_up = self._augmentaion(MS_image, PAN_image, reference, MS_image_up)    ### \\- 2023-0315 rewrite -\\
            #print("HSI_datasets.py --> pavia_dataset: augmentation has done!!! ")	### \\ -2023-0315 add - \\
        if self.split == "train" and index == len(self.files[self.split]) - 1:
            np.random.shuffle(self.files[self.split])

        #return image_dict, MS_image, PAN_image, reference
        return image_dict, MS_image, PAN_image, reference, MS_image_up ### \\- 2023-0315 rewrite -\\

    
    def __getitem__(self, index):

        #image_dict, MS_image, PAN_image, reference = self.getHSIdata(index)
        image_dict, MS_image, PAN_image, reference, MS_image_up = self.getHSIdata(index)    ### \\- 2023-0315 rewrite -\\
        #return image_dict, MS_image, PAN_image, reference
        return image_dict, MS_image, PAN_image, reference, MS_image_up ### \\- 2023-0315 rewrite -\\


