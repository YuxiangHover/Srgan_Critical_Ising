# dataset.py 

import torch
from torch.utils.data import Dataset
import numpy as np
import torch.nn.functional as F
import glob
import os 

class IsingDataset(Dataset):
    def __init__(self, data_pattern, scale_factor=4):
        
       
        self.file_list = glob.glob(data_pattern) 
        self.file_list.sort() 

        print(f" {len(self.file_list)} have been found")
        
        if not self.file_list:
             raise FileNotFoundError(...)
             
        all_data_list = []
        for file_path in self.file_list:
            try:
                data = np.load(file_path).astype(np.float32)
                
                if data.ndim == 3 and data.shape[1] == 512 and data.shape[2] == 512:
                    all_data_list.append(data)
                else:
                    print(f"Warning: {file_path} shape ({data.shape})  does not match, skip")
            except Exception as e:
                print(f"Warning: fail to load{file_path}  ({e})，skip.")


        self.data = np.concatenate(all_data_list, axis=0)
        self.scale_factor = scale_factor
        self.total_configs = len(self.data)
        print(f"Number of configs: {self.total_configs}")
        
    def __len__(self):
        return self.total_configs

    def __getitem__(self, idx):
        hr_config_numpy = self.data[idx] 
        
        # (512, 512) -> (1, 512, 512)
        hr_img = torch.from_numpy(hr_config_numpy).unsqueeze(0)
        
        # 2.  LR (Low Res) 
        lr_mean = F.avg_pool2d(hr_img, kernel_size=self.scale_factor, stride=self.scale_factor)
        
        # 3.  (Sign)  LR {-1, 1}
        lr_img = torch.sign(lr_mean)
        lr_img[lr_img == 0] = 1.0 
        
        k = np.random.randint(0, 4) 
        flip_h = np.random.rand() > 0.5 
        flip_v = np.random.rand() > 0.5 

        lr_img = torch.rot90(lr_img, k=k, dims=[1, 2])
        hr_img = torch.rot90(hr_img, k=k, dims=[1, 2])
        if flip_h:
            lr_img = torch.flip(lr_img, [2]) 
            hr_img = torch.flip(hr_img, [2])
        if flip_v:
            lr_img = torch.flip(lr_img, [1]) 
            hr_img = torch.flip(hr_img, [1])
            

        return lr_img, hr_img
