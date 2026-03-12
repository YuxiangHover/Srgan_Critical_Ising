# dataset.py (最终稳定版本：一次性加载 5000 配置，无 load_file_if_needed)

import torch
from torch.utils.data import Dataset
import numpy as np
import torch.nn.functional as F
import glob
import os 

# dataset.py (Linux 90GB RAM 优化版)

import torch
from torch.utils.data import Dataset
import numpy as np
import torch.nn.functional as F
import glob
import os 

class IsingDataset(Dataset):
    def __init__(self, data_pattern, scale_factor=4):
        
        # 1. 找到所有匹配的文件路径 (查找全部 20 个文件)
        self.file_list = glob.glob(data_pattern) 
        self.file_list.sort() 

        # # === 快速验证修改点：只使用前 2 个文件 (1000个配置) ===
        # NUM_FILES_FOR_TEST = 2
        # self.file_list = self.file_list[:NUM_FILES_FOR_TEST] 
        # print(f"【快速验证模式】仅使用前 {NUM_FILES_FOR_TEST} 个文件加载数据。")


        
        # --- 不进行文件数量限制，使用全部文件 ---
        print(f"找到 {len(self.file_list)} 个数据文件，将全部加载到内存。")
        
        if not self.file_list:
             raise FileNotFoundError(...)
             
        # === 核心：一次性加载并连接所有数据 (~10.48 GB) ===
        all_data_list = []
        for file_path in self.file_list:
            try:
                # 尝试加载文件
                data = np.load(file_path).astype(np.float32)
                
                # 针对您之前的错误，进行一次形状检查
                if data.ndim == 3 and data.shape[1] == 512 and data.shape[2] == 512:
                    all_data_list.append(data)
                else:
                    print(f"警告: 文件 {file_path} 形状不符 ({data.shape})，跳过该文件。")
            except Exception as e:
                # 捕捉文件损坏错误
                print(f"警告: 文件 {file_path} 读取失败或损坏 ({e})，跳过该文件。")


        self.data = np.concatenate(all_data_list, axis=0)
        self.scale_factor = scale_factor
        self.total_configs = len(self.data)
        print(f"总配置数量: {self.total_configs}")
        
    def __len__(self):
        return self.total_configs

    def __getitem__(self, idx):
        # 1. 获取 HR (High Res) 配置
        # === 修正 1：直接从内存中的 self.data 读取数据 ===
        hr_config_numpy = self.data[idx] 
        
        # 增加通道维度: (512, 512) -> (1, 512, 512)
        hr_img = torch.from_numpy(hr_config_numpy).unsqueeze(0)
        
        # 2. 生成 LR (Low Res) - 块平均降采样
        lr_mean = F.avg_pool2d(hr_img, kernel_size=self.scale_factor, stride=self.scale_factor)
        
        # 3. 符号化 (Sign) 得到最终 LR 配置 {-1, 1}
        lr_img = torch.sign(lr_mean)
        lr_img[lr_img == 0] = 1.0 
        
        # --- 随机数据增强 (旋转和翻转) ---
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