import os
import glob
import torch
import numpy as np
import time
import gc
from model import Generator

# ==================== 终极保守配置 ====================
INPUT_ROOT = r"D:\Code\final_srgan\configurations"
OUTPUT_ROOT = r"D:\Code\final_srgan\SRGAN_output"
CHECKPOINT_PATH = "checkpoint_epoch_95.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 强制限制 CPU 只用 1 个核，防止电脑卡死
torch.set_num_threads(1) 

# TARGET_FOLDERS = ["128x128", "256x256", "512x512"]
TARGET_FOLDERS = ["64x64"]
# ======================================================

def load_model():
    print(f"🚀 加载模型: {CHECKPOINT_PATH}")
    # 显存清理
    torch.cuda.empty_cache()
    
    netG = Generator(noise_std=0.1).to(DEVICE)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    if 'netG_state_dict' in checkpoint:
        netG.load_state_dict(checkpoint['netG_state_dict'])
    else:
        netG.load_state_dict(checkpoint)
    netG.eval()
    return netG

def process_one_file_ultra_safe(netG, file_path, output_dir):
    file_name = os.path.basename(file_path)
    save_path = os.path.join(output_dir, file_name)

    # 1. 如果文件已存在，直接跳过 (断点续传)
    if os.path.exists(save_path):
        # 简单的检查：如果文件大小不对（可能是坏的），则重跑
        # 这里简单起见，只要存在就跳过
        print(f"   ⚠️ 跳过已存在文件: {file_name}")
        return

    # 2. 读取输入 (加载到内存，因为1个文件也就几百M，这步应该没问题)
    try:
        # data shape: (500, H, W)
        data_cpu = np.load(file_path).astype(np.float32)
    except Exception as e:
        print(f"❌ 读取错误: {e}")
        return

    num_imgs = data_cpu.shape[0]
    h = data_cpu.shape[1]
    w = data_cpu.shape[2]
    
    # 预分配输出数组 (int8 极省内存)
    # (500, 4H, 4W)
    output_data = np.zeros((num_imgs, h*4, w*4), dtype=np.int8)

    # 3. 逐张处理 (Loop 500 次)
    with torch.no_grad():
        for i in range(num_imgs):
            # 取出 1 张
            img = data_cpu[i] # (H, W)
            
            # 增加维度 (1, 1, H, W)
            img_tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(DEVICE)
            
            # 推理
            sr_out = netG(img_tensor)
            
            # 二值化
            sr_binary = torch.sign(sr_out)
            sr_binary[sr_binary == 0] = 1.0
            
            # 转回 CPU 并填入结果数组
            # squeeze 之后是 (4H, 4W)
            res_numpy = sr_binary.cpu().numpy().squeeze().astype(np.int8)
            output_data[i] = res_numpy
            
            # === 关键：立即销毁变量 ===
            del img_tensor, sr_out, sr_binary, res_numpy
            # 每处理 50 张清理一次显存，不要太频繁否则变慢，也不要太久
            if i % 50 == 0:
                torch.cuda.empty_cache()
    
    # 4. 保存文件
    try:
        np.save(save_path, output_data)
        print(f"   ✅ 完成: {file_name}")
    except Exception as e:
        print(f"❌ 保存失败: {e}")

    # 5. 文件级清理
    del data_cpu
    del output_data
    gc.collect() # 强制回收内存
    torch.cuda.empty_cache()

def main():
    if not os.path.exists(OUTPUT_ROOT):
        os.makedirs(OUTPUT_ROOT)
    
    netG = load_model()
    
    for folder_name in TARGET_FOLDERS:
        input_dir = os.path.join(INPUT_ROOT, folder_name)
        output_dir = os.path.join(OUTPUT_ROOT, folder_name)
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        all_files = glob.glob(os.path.join(input_dir, "*.npy"))
        print(f"\n📂 开始处理目录: {folder_name} (共 {len(all_files)} 个文件)")
        
        for idx, fp in enumerate(all_files):
            process_one_file_ultra_safe(netG, fp, output_dir)
            
            # 每处理完一个文件，打印一次内存提示，防止你焦虑
            if (idx + 1) % 1 == 0:
                print(f"   进度: {idx + 1}/{len(all_files)}")
                # 稍微休息 0.1 秒，把 CPU 让给操作系统，防止卡死
                time.sleep(0.1)

if __name__ == "__main__":
    main()