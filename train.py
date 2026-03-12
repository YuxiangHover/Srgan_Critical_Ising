import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from dataset import IsingDataset
from model import Generator, Discriminator
import matplotlib.pyplot as plt
import numpy as np
import time

# === 配置 ===
BATCH_SIZE = 16
EPOCHS = 100
LR_G = 1e-4
LR_D = 5e-6
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# === 物理约束损失 ===
class MagnetizationLoss(nn.Module):
    def __init__(self):
        super(MagnetizationLoss, self).__init__()
        self.l1 = nn.L1Loss()
        
    def forward(self, sr, hr):
        mag_sr = torch.mean(sr, dim=[1, 2, 3])
        mag_hr = torch.mean(hr, dim=[1, 2, 3])
        return self.l1(mag_sr, mag_hr)

def train():
    # 1. 加载完整数据集
    DATA_DIR = "ising_configurations_512x512"
    FILE_PATTERN = "ising_Size512_*.npy" 
    DATA_PATTERN = os.path.join(DATA_DIR, FILE_PATTERN)
    
    print("正在加载数据集...")
    full_dataset = IsingDataset(DATA_PATTERN)
    total_size = len(full_dataset)
    print(f"数据集加载完毕，共 {total_size} 个配置。")

    # 2. 划分数据集
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(SEED)
    )

    print(f"数据集划分: 训练集 {len(train_dataset)}, 验证集 {len(val_dataset)}, 测试集 {len(test_dataset)}")

    # 3. DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    
    # 4. 模型初始化
    # 如果你想增加生成器的随机性，可以在这里设置 noise_std，例如 Generator(noise_std=0.05)
    
    # === 修改前 ===
    # netG = Generator().to(DEVICE) # 默认 noise_std=0.01
    
    # === 修改后 ===
    # 强行注入更强的随机噪声，给生成器制造噪点的“种子”
    netG = Generator(noise_std=0.1).to(DEVICE)
    netD = Discriminator().to(DEVICE)
    
    optimizerG = optim.Adam(netG.parameters(), lr=LR_G, betas=(0.5, 0.999))
    optimizerD = optim.Adam(netD.parameters(), lr=LR_D, betas=(0.5, 0.999))
    
    criterion_pixel = nn.L1Loss() 
    criterion_adv = nn.BCELoss()
    criterion_mag = MagnetizationLoss()

    print(f"开始训练! Device: {DEVICE}")
    
    WARMUP_EPOCHS = 5 # 前5轮只训练G，不训练D
    best_val_loss = float('inf') 
    
    G_loss_history = []
    D_loss_history = []
    Val_loss_history = [] 

    start_time_train = time.time()

    for epoch in range(EPOCHS):
        start_time_epoch = time.time()
        
        # =========================
        #      Train (训练阶段)
        # =========================
        netG.train()
        netD.train()
        
        train_loss_g_accum = 0.0
        train_loss_d_accum = 0.0 # 新增 D Loss 累积
        
        for i, (lr_imgs, hr_imgs) in enumerate(train_loader):
            lr_imgs = lr_imgs.to(DEVICE)
            hr_imgs = hr_imgs.to(DEVICE)
            batch_size = lr_imgs.size(0)
            
            real_label = torch.full((batch_size,), 0.9).to(DEVICE)
            fake_label = torch.zeros(batch_size).to(DEVICE)
            
            # --- 1. Train Generator ---
            netG.zero_grad()
            fake_hr = netG(lr_imgs)
            
            loss_pixel = criterion_pixel(fake_hr, hr_imgs)
            loss_mag = criterion_mag(fake_hr, hr_imgs)
            
            if epoch < WARMUP_EPOCHS:
                # Warmup: 仅 L1 Loss
                loss_G = loss_pixel + 0.1 * loss_mag
                loss_G.backward()
                optimizerG.step()
            else:
                # Adversarial: 加入 D 的反馈
                # === 关键修改：提高 Adv 权重，降低 Pixel 权重，逼迫生成细节 ===
                output_fake_for_G = netD(fake_hr)
                loss_adv = criterion_adv(output_fake_for_G, real_label)
                
                # 旧权重: loss_pixel + 0.005 * loss_adv ...
                # 新权重: 0.5 * loss_pixel + 0.05 * loss_adv ...
                # === 修改前 ===
                # loss_G = 0.5 * loss_pixel + 0.05 * loss_adv + 0.1 * loss_mag
                
                # === 修改后：激进策略 ===
                # 1. Pixel 权重降为 0.01 (只要大轮廓对就行，细节别管)
                # 2. Adv 权重升为 0.5 (必须骗过判别器，判别器要求有噪点)
                # 3. Mag 保持 0.1 (防止全黑/全白)
                loss_G = 0.5 * loss_pixel + 0.05 * loss_adv + 0.1 * loss_mag

                loss_G.backward()
                optimizerG.step()

            train_loss_g_accum += loss_G.item()

            # --- 2. Train Discriminator (Warmup 阶段跳过) ---
            if epoch >= WARMUP_EPOCHS:
                netD.zero_grad()
                output_real = netD(hr_imgs)
                loss_real = criterion_adv(output_real, real_label)
                
                # detach() 防止梯度回传给 G
                output_fake = netD(fake_hr.detach())
                loss_fake = criterion_adv(output_fake, fake_label)
                
                loss_D = (loss_real + loss_fake) / 2
                loss_D.backward()
                optimizerD.step()
                
                train_loss_d_accum += loss_D.item()
            else:
                # Warmup 期间 D Loss 为 0
                train_loss_d_accum += 0.0

        # 计算平均 Loss
        avg_train_loss_g = train_loss_g_accum / len(train_loader)
        avg_train_loss_d = train_loss_d_accum / len(train_loader)
        
        G_loss_history.append(avg_train_loss_g)
        D_loss_history.append(avg_train_loss_d)

        # =========================
        #    Validation (验证阶段)
        # =========================
        netG.eval() 
        val_loss_accum = 0.0
        
        with torch.no_grad(): 
            for lr_val, hr_val in val_loader:
                lr_val = lr_val.to(DEVICE)
                hr_val = hr_val.to(DEVICE)
                sr_val = netG(lr_val)
                
                # 验证集只看生成质量 (Pixel + Mag)
                v_pixel = criterion_pixel(sr_val, hr_val)
                v_mag = criterion_mag(sr_val, hr_val)
                
                # 保持验证集的评判标准不变，或者也可以对齐训练集的权重
                val_loss = v_pixel + 0.1 * v_mag
                val_loss_accum += val_loss.item()
        
        avg_val_loss = val_loss_accum / len(val_loader)
        Val_loss_history.append(avg_val_loss)

        # =========================
        #    Logging & Saving
        # =========================
        end_time_epoch = time.time()
        time_epoch = end_time_epoch - start_time_epoch
        
        # 格式化输出字符串
        if epoch < WARMUP_EPOCHS:
            d_loss_str = "0.000 (Warmup)"
        else:
            d_loss_str = f"{avg_train_loss_d:.5f}"

        print(f"[Epoch {epoch}/{EPOCHS}] Loss G: {avg_train_loss_g:.5f} | Loss D: {d_loss_str} | Val Loss: {avg_val_loss:.5f} | Time: {time_epoch:.0f}s")

        # 保存最优模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(netG.state_dict(), "ising_srgan_G_best.pth")
            print(f"  >>> 发现更优模型 (Val Loss: {best_val_loss:.5f})，已保存 Best Checkpoint。")

        # 可视化
        if epoch % 1 == 0:
            with torch.no_grad():
                lr_view, hr_view = next(iter(val_loader))
                lr_view = lr_view.to(DEVICE)
                sr_view = netG(lr_view)
                
                sr_show = sr_view[0].cpu().squeeze().numpy()
                hr_show = hr_view[0].cpu().squeeze().numpy()
                lr_show = lr_view[0].cpu().squeeze().numpy()
                
                plt.figure(figsize=(12, 4))
                plt.subplot(1, 3, 1); plt.imshow(lr_show, cmap='gray'); plt.title("Input (Val)")
                plt.subplot(1, 3, 2); plt.imshow(sr_show, cmap='gray', vmin=-1, vmax=1); plt.title(f"SR (Val) Ep:{epoch}")
                plt.subplot(1, 3, 3); plt.imshow(hr_show, cmap='gray'); plt.title("GT (Val)")
                plt.savefig(f"epoch_{epoch}_val.png")
                plt.close()

                # === [补丁] 保存每个 Epoch 的 Checkpoint (方便回溯) ===
        # 建议每 5 个 epoch 保存一次，或者是崩溃高发期保存
        # 如果硬盘空间足够，也可以去掉 if epoch % 5 == 0 的限制
                if epoch % 1 == 0: 
                    checkpoint_state = {
                        'epoch': epoch,
                        'netG_state_dict': netG.state_dict(),
                        'netD_state_dict': netD.state_dict(),
                        'optimizerG_state_dict': optimizerG.state_dict(),
                        'optimizerD_state_dict': optimizerD.state_dict(),
                        'loss_G': avg_train_loss_g,
                        'loss_D': avg_train_loss_d,
                        'loss_val': avg_val_loss
                    }
                    torch.save(checkpoint_state, f"checkpoint_epoch_{epoch}.pth")
                    print(f"Checkpoint saved: epoch_{epoch}")

    # =========================
    #      Test (最终测试)
    # =========================
    print("\n训练结束，开始在 [测试集] 上进行最终评估...")
    netG.load_state_dict(torch.load("ising_srgan_G_best.pth"))
    netG.eval()
    
    total_l1 = 0.0
    total_mse = 0.0
    
    with torch.no_grad():
        for lr_test, hr_test in test_loader:
            lr_test = lr_test.to(DEVICE)
            hr_test = hr_test.to(DEVICE)
            sr_test = netG(lr_test)
            total_l1 += nn.L1Loss()(sr_test, hr_test).item()
            total_mse += nn.MSELoss()(sr_test, hr_test).item()
            
    avg_test_l1 = total_l1 / len(test_loader)
    avg_test_mse = total_mse / len(test_loader)
    
    print(f"=== 最终测试结果 (Test Set) ===")
    print(f"Average L1 Loss (MAE): {avg_test_l1:.6f}")
    print(f"Average MSE Loss:      {avg_test_mse:.6f}")
    
    # 绘制详细 Loss 曲线
    plt.figure()
    plt.plot(G_loss_history, label='Train Loss G')
    plt.plot(D_loss_history, label='Train Loss D') # 现在可以看到 D 的变化了
    plt.plot(Val_loss_history, label='Val Loss (Pixel+Mag)')
    plt.legend()
    plt.title("Training Loss Details")
    plt.savefig("loss_curve.png")

    print("所有流程完成。")

if __name__ == "__main__":
    train()