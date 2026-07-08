from torchvision import transforms
from utils import *

from datetime import datetime

class configs:
    """
    the config of training setting.
    """
    
    network = 'vmunet-v2'
    # network = 'unet'少加了batch
    # network = 'Unet3plus'
    # network = 'PSPNet'
    # network = 'vmunet-v2'
    # network = 'VMUnet3plus'
    # network = 'unetplse'
    




    datasets = 'Pterygium82' 
    if datasets == 'isic18':
        data_path = './data/isic2018/'
    elif datasets == 'isic17':
        data_path = './data/isic2017/'
    elif datasets == 'Pterygium82':
        data_path = './data/Pterygium82/'
    elif datasets == 'source811':
        data_path = './data/source811/' 
    elif datasets == 'edema811':
        data_path = './data/edema811/' 
    elif datasets == 'muscles811':
        data_path = './data/muscles811/' 
    elif datasets == 'background811':
        data_path = './data/background811/' 
    else:
        raise Exception('datasets in not right!')




    # ===== Optimizer: 修改此处选择优化器（--opt CLI 可覆盖）=====
    opt = 'AdamW'
    assert opt in ['Adadelta', 'Adagrad', 'Adam', 'AdamW', 'Adamax', 'ASGD', 'RMSprop', 'Rprop', 'SGD'], 'Unsupported optimizer!'

    # 所有优化器参数无条件定义，保证 --opt CLI 覆盖后不会 AttributeError
    lr           = 0.001          # 学习率（--lr 可覆盖）
    betas        = (0.9, 0.999)   # Adam/AdamW/Adamax
    eps          = 1e-8           # 数值稳定项
    weight_decay = 1e-2           # 权重衰减
    amsgrad      = False          # AMSGrad 变体
    momentum     = 0.9            # SGD / RMSprop 动量
    dampening    = 0              # SGD 阻尼
    nesterov     = False          # SGD Nesterov
    alpha        = 0.99           # RMSprop 平滑常数
    centered     = False          # RMSprop 归一化
    rho          = 0.9            # Adadelta
    lr_decay     = 0              # Adagrad
    lambd        = 1e-4           # ASGD 衰减项
    t0           = 1e6            # ASGD 开始平均时间点
    etas         = (0.5, 1.2)     # Rprop 乘法因子
    step_sizes   = (1e-6, 50)     # Rprop 步长范围

    # ===== Scheduler: 修改此处选择调度器（--sch CLI 可覆盖）=====
    sch = 'CosineAnnealingLR'
    assert sch in ['StepLR', 'MultiStepLR', 'ExponentialLR', 'CosineAnnealingLR', 'ReduceLROnPlateau',
                   'CosineAnnealingWarmRestarts', 'WP_MultiStepLR', 'WP_CosineLR'], 'Unsupported scheduler!'

    # 所有调度器参数无条件定义，保证 --sch CLI 覆盖后不会 AttributeError
    epochs           = 200            # 总训练轮数（WP_CosineLR 需要此参数）
    step_size        = 40             # StepLR
    milestones       = [60, 120, 160] # MultiStepLR / WP_MultiStepLR
    gamma            = 0.1            # 衰减因子
    last_epoch       = -1             # 通用
    T_max            = 200            # CosineAnnealingLR（= epochs，单次完整余弦）
    eta_min          = 1e-6           # 学习率下限
    T_0              = 50             # CosineAnnealingWarmRestarts 首次重启周期
    T_mult           = 2              # 重启后周期倍增
    mode             = 'min'          # ReduceLROnPlateau
    factor           = 0.5            # ReduceLROnPlateau 衰减因子
    patience         = 10             # ReduceLROnPlateau 容忍轮数
    threshold        = 1e-4           # ReduceLROnPlateau 阈值
    threshold_mode   = 'rel'          # ReduceLROnPlateau 阈值模式
    cooldown         = 0              # ReduceLROnPlateau 冷却轮数
    min_lr           = 1e-7           # ReduceLROnPlateau 学习率下限
    warm_up_epochs   = 20             # WP_CosineLR / WP_MultiStepLR Warmup 轮数


if __name__ == '__main__':
    print(f"opt={configs.opt}, sch={configs.sch}")