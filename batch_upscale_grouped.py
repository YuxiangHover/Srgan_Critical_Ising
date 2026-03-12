import os
import glob
import torch
import numpy as np
import time
import gc
from model import Generator

INPUT_ROOT = r"configurations"
OUTPUT_ROOT = r"SRGAN_output"
CHECKPOINT_PATH = "checkpoint.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# TARGET_FOLDERS = ["128x128", "256x256", "512x512"]
TARGET_FOLDERS = ["64x64"]
# ======================================================

def load_model():
    print(f"loading: {CHECKPOINT_PATH}")

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
    
    if os.path.exists(save_path):
        return

    try:
        # data shape: (500, H, W)
        data_cpu = np.load(file_path).astype(np.float32)
    except Exception as e:
        print(fail to load: {e}")
        return

    num_imgs = data_cpu.shape[0]
    h = data_cpu.shape[1]
    w = data_cpu.shape[2]
    
    # (500, 4H, 4W)
    output_data = np.zeros((num_imgs, h*4, w*4), dtype=np.int8)

    with torch.no_grad():
        for i in range(num_imgs):
            img = data_cpu[i] # (H, W)
            
            img_tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(DEVICE)

            sr_out = netG(img_tensor)
            
            sr_binary = torch.sign(sr_out)
            sr_binary[sr_binary == 0] = 1.0

            # squeeze 之后是 (4H, 4W)
            res_numpy = sr_binary.cpu().numpy().squeeze().astype(np.int8)
            output_data[i] = res_numpy
            
            del img_tensor, sr_out, sr_binary, res_numpy
            if i % 50 == 0:
                torch.cuda.empty_cache()

    try:
        np.save(save_path, output_data)
        print(f" Done : {file_name}")
    except Exception as e:
        print(f"Fail to save: {e}")

    del data_cpu
    del output_data
    gc.collect() 
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
        print(f"\n saving to: {folder_name} ( {len(all_files)} files)")
        
        for idx, fp in enumerate(all_files):
            process_one_file_ultra_safe(netG, fp, output_dir)
            
            if (idx + 1) % 1 == 0:
                print(f"   processing: {idx + 1}/{len(all_files)}")
                time.sleep(0.1)

if __name__ == "__main__":

    main()
