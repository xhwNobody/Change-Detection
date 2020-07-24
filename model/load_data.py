import os
import torch
import numpy as np
from torch.utils.data.dataset import Dataset

class SAR_Dataset(Dataset):
    def __init__(self, imgPath, labPath):
        self.imgNameList = os.listdir(imgPath)
        self.imgPath = imgPath
        self.labPath = labPath

    def __getitem__(self, index):

        img = np.load(self.imgPath + self.imgNameList[index])
        label = np.load(self.labPath + self.imgNameList[index])

        img = np.swapaxes(np.swapaxes(img, 0, 2), 1, 2)
        img = torch.from_numpy(img).type(torch.FloatTensor)
        label = torch.from_numpy(label).type(torch.FloatTensor)

        return img, label

    def __len__(self):
        return len(self.imgNameList)