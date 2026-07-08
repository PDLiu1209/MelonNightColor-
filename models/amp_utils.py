import numpy as np
import torch

def extract_amp(img_np):
    """提取图像的幅度谱"""
    fft = np.fft.fft2(img_np, axes=(-2, -1))
    amp_np = np.abs(fft)
    return amp_np

def mutate(amp_src, amp_trg, L=0.1):
    """在频域中心区域用目标幅度替换源幅度"""
    a_src = np.fft.fftshift(amp_src, axes=(-2, -1))
    a_trg = np.fft.fftshift(amp_trg, axes=(-2, -1))
    
    h = a_src.shape[1]
    w = a_src.shape[2]

    b = int(np.floor(np.amin((h, w)) * L))
    c_h = int(np.floor(h / 2.0))
    c_w = int(np.floor(w / 2.0))

    h1 = c_h - b
    h2 = c_h + b + 1
    w1 = c_w - b
    w2 = c_w + b + 1

    a_src[:, h1:h2, w1:w2] = a_trg[:, h1:h2, w1:w2]
    a_src = np.fft.ifftshift(a_src, axes=(-2, -1))
    return a_src

def normalize(src_img, amp_trg, L=0.1):
    """使用目标幅度谱对源图像进行归一化"""
    fft_src_np = np.fft.fft2(src_img, axes=(-2, -1))
    amp_src = np.abs(fft_src_np)
    pha_src = np.angle(fft_src_np)
    amp_src_ = mutate(amp_src, amp_trg, L=L)
    fft_src_ = amp_src_ * np.exp(1j * pha_src)
    src_in_trg = np.fft.ifft2(fft_src_, axes=(-2, -1))
    src_in_trg_real = np.real(src_in_trg)
    return src_in_trg_real

def process(x, running_amp, momentum=0.1, fix_amp=False):
    """
    处理批次数据的幅度谱归一化
    
    Args:
        x: 输入张量 (B, C, H, W)
        running_amp: 运行时幅度谱 (C, H, W)
        momentum: 动量参数
        fix_amp: 是否固定幅度谱
    
    Returns:
        处理后的x和更新的running_amp
    """
    # 如果输入是torch tensor，转换为numpy
    if isinstance(x, torch.Tensor):
        x_np = x.detach().cpu().numpy()
        is_torch = True
        device = x.device
        dtype = x.dtype
    else:
        x_np = x
        is_torch = False
    
    if isinstance(running_amp, torch.Tensor):
        running_amp_np = running_amp.detach().cpu().numpy()
        running_amp_is_torch = True
        running_amp_device = running_amp.device
        running_amp_dtype = running_amp.dtype
    else:
        running_amp_np = running_amp
        running_amp_is_torch = False
    
    B, C, H, W = x_np.shape
    
    # 确保running_amp的形状正确
    if running_amp_np.shape != (C, H, W):
        running_amp_np = np.zeros((C, H, W), dtype=np.float32)
    
    amp_list = np.zeros((B, C, H, W), dtype=np.float32)
    
    if not fix_amp:    
        for idx in range(B):
            amp_np = extract_amp(x_np[idx])
            amp_list[idx, :, :, :] = amp_np
        amp_avg = np.mean(amp_list, axis=0)
        
        if np.sum(running_amp_np) == 0:
            running_amp_np = amp_avg
        else:
            running_amp_np = running_amp_np * (1 - momentum) + amp_avg * momentum
    
    # 对每个样本进行归一化
    for idx in range(B):
        x_np[idx] = normalize(x_np[idx], running_amp_np[:3, ...], L=0)
    
    # 转换回原始格式
    if is_torch:
        x = torch.from_numpy(x_np).to(device).to(dtype)
    else:
        x = x_np
        
    if running_amp_is_torch:
        running_amp = torch.from_numpy(running_amp_np).to(running_amp_device).to(running_amp_dtype)
    else:
        running_amp = running_amp_np

    return x, running_amp
    