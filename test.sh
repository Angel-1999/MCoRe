#!/usr/bin/env sh
mkdir -p logs
DN="w_DN" # 使用Dive Number
nam="rny002_gsm_gru_0127_1942"  # 测试使用的已训练模型文件名，在experiments/FineDiving/w_DN文件夹中
config_name="test_$DN/$nam"
log_name="LOG_Test_$nam_duration"
exp_name="./experiments/$1/$DN/$nam/last.pth"

CUDA_VISIBLE_DEVICES=$4 python3 -u main.py --benchmark $1 --prefix $config_name --test --ckpts $exp_name --feature_arch $2 --temporal_arch $3 2>&1|tee logs/$log_name.log

# 测试运行指令e.g: bash test.sh FineDiving rny002_gsm gru 0 [#指定gpu=0]

