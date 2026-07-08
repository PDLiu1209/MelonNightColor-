#!/bin/bash

#SBATCH --job-name=seg_model               # 作业名称
#SBATCH --cpus-per-task=4                  # CPU 核心数
#SBATCH --gres=gpu:4090:1                  # 请求 1 个 4090
#SBATCH --output=slurm-%j-out             # 输出文件名，%j 替换为作业ID
source ~/anaconda3/etc/profile.d/conda.sh
conda activate LTBSoftmax

JOB_ID=$SLURM_JOB_ID
OUTPUT_FILE1="slurm-${JOB_ID}-out"
echo "The job id is : $JOB_ID"
echo "Output file: $OUTPUT_FILE1"

START_TIME=$(date +%s)
echo "Job started at: $(date)" >> $OUTPUT_FILE1

echo "Using GPUs:"
nvidia-smi >> $OUTPUT_FILE1

sinfo -N -O nodelist,gres,statelong
cat /proc/cpuinfo | grep 'model name' | uniq

#数据集名称
#       data_red 
#       data_blue
#       data_green
#       data_cyan
#       data_purple
#       data_yellow
#       data_white
#       data_red_blue
#       data_red_green
#       data_green_blue
#       data_red_green_blue
# ============================================================
# 固定超参数：AdamW + WP_CosineLR + lr=0.001 + warmup=20
#             wd=5e-4 + CEDL loss + dice_weight=0.7
# 使用方法：取消注释想运行的模型命令，其余保持注释。
# ============================================================
# ============================================================
# 【基础模型】
# ============================================================
# mA: Unet（基准）
#python /public/home/liupengdong2024/Paper_Two/New_Seg_Model_Code/train.py --network Unet --data data_red_green_blue --opt AdamW --sch WP_CosineLR --lr 0.001 --warmup 20 --wd 5e-4 --loss CEDL --dice-weight 0.7 >> $OUTPUT_FILE1 2>&1

# mB: PSPNet
#python /public/home/liupengdong2024/Paper_Two/New_Seg_Model_Code/train.py --network PSPNet --data data_red_green_blue --opt AdamW --sch WP_CosineLR --lr 0.001 --warmup 20 --wd 5e-4 --loss CEDL --dice-weight 0.7 >> $OUTPUT_FILE1 2>&1

# mC: unet3up（UNet3+类结构）
#python /public/home/liupengdong2024/Paper_Two/New_Seg_Model_Code/train.py --network unet3up --data data_red_green_blue --opt AdamW --sch WP_CosineLR --lr 0.001 --warmup 20 --wd 5e-4 --loss CEDL --dice-weight 0.7 >> $OUTPUT_FILE1 2>&1

# mD: SwinUnet
#python /public/home/liupengdong2024/Paper_Two/New_Seg_Model_Code/train.py --network SwinUnet --data data_red_green_blue --opt AdamW --sch WP_CosineLR --lr 0.001 --warmup 20 --wd 5e-4 --loss CEDL --dice-weight 0.7 >> $OUTPUT_FILE1 2>&1

# mE: VMUNet
#python /public/home/liupengdong2024/Paper_Two/New_Seg_Model_Code/train.py --network VMUNet --data data_red_green_blue --opt AdamW --sch WP_CosineLR --lr 0.001 --warmup 20 --wd 5e-4 --loss CEDL --dice-weight 0.7 >> $OUTPUT_FILE1 2>&1

# mF: DenseNet
#python /public/home/liupengdong2024/Paper_Two/New_Seg_Model_Code/train.py --network DenseNet --data data_red_green_blue --opt AdamW --sch WP_CosineLR --lr 0.001 --warmup 20 --wd 5e-4 --loss CEDL --dice-weight 0.7 >> $OUTPUT_FILE1 2>&1

# mG: DenseUNet
#python /public/home/liupengdong2024/Paper_Two/New_Seg_Model_Code/train.py --network DenseUNet --data data_red_green_blue --opt AdamW --sch WP_CosineLR --lr 0.001 --warmup 20 --wd 5e-4 --loss CEDL --dice-weight 0.7 >> $OUTPUT_FILE1 2>&1

# mH: DeepLab
#python /public/home/liupengdong2024/Paper_Two/New_Seg_Model_Code/train.py --network DeepLab --data data_red_green_blue --opt AdamW --sch WP_CosineLR --lr 0.001 --warmup 20 --wd 5e-4 --loss CEDL --dice-weight 0.7 >> $OUTPUT_FILE1 2>&1

# mI: H_vmunet
#python /public/home/liupengdong2024/Paper_Two/New_Seg_Model_Code/train.py --network H_vmunet --data data_red_green_blue --opt AdamW --sch WP_CosineLR --lr 0.001 --warmup 20 --wd 5e-4 --loss CEDL --dice-weight 0.7 >> $OUTPUT_FILE1 2>&1

# ============================================================
# 【改进模型】（基于 UNet 的改进与消融组合）
# ============================================================
# mJ: UNet_BN（UNet + BatchNorm）
#python /public/home/liupengdong2024/Paper_Two/New_Seg_Model_Code/train.py --network UNet_BN --data data_purple --opt AdamW --sch WP_CosineLR --lr 0.001 --warmup 20 --wd 5e-4 --loss CEDL --dice-weight 0.7 >> $OUTPUT_FILE1 2>&1

# mL: UNet_Decoder_Pyramid（UNet + 解码器金字塔融合）
#python /public/home/liupengdong2024/Paper_Two/New_Seg_Model_Code/train.py --network UNet_Decoder_Pyramid --data data_red_green_blue --opt AdamW --sch WP_CosineLR --lr 0.001 --warmup 20 --wd 5e-4 --loss CEDL --dice-weight 0.7 >> $OUTPUT_FILE1 2>&1

# mN: UNet_BN_DecoderPyramid（UNet + BN + 解码器金字塔）
python /public/home/liupengdong2024/Paper_Two/New_Seg_Model_Code/train.py --network UNet_BN_DecoderPyramid --data data_white --opt AdamW --sch WP_CosineLR --lr 0.001 --warmup 20 --wd 5e-4 --loss CEDL --dice-weight 0.7 >> $OUTPUT_FILE1 2>&1

#数据集名称
#       data_red 
#       data_blue
#       data_green
#       data_cyan
#       data_purple
#       data_yellow
#       data_white
#       data_red_blue
#       data_red_green
#       data_green_blue
#       data_red_green_blue

END_TIME=$(date +%s)
echo "Job ended at: $(date)" >> $OUTPUT_FILE1
RUN_TIME=$(($END_TIME - $START_TIME))
echo "Job ran for: $RUN_TIME seconds" >> $OUTPUT_FILE1
