@echo off
:: 该脚本将训练 VisA 和 mvtec 数据集的多尺度模型
:: 将脚本保存为 .bat 文件，例如 `train.bat`

:: 设置 CUDA_VISIBLE_DEVICES 为当前可用的 GPU 设备号
set device=0

:: 定义日志文件路径
set LOG=%save_dir%"res.log"
echo %LOG%

:: 模型深度和上下文设置
set depth=9
set n_ctx=12
set t_n_ctx=4

:: 训练 VisA 数据集的循环
for i in "%depth%".split() do ^
    for j in "%n_ctx%".split() do ^
        set base_dir=%depth%_%n_ctx%_%t_n_ctx%_multiscale_visa
        set save_dir=..\checkpoints\%base_dir%
        CUDA_VISIBLE_DEVICES=%device% cmd /c call test.py ^
            --dataset visa ^
            --data_path "%remote-home%\iot_zhouqihang\data\Visa" ^
            --save_path %save_dir%\zero_shot ^
            --checkpoint_path %save_dir%epoch_15.pth ^
            --features_list 6 12 18 24 ^
            --image_size 518 ^
            --depth %depth% ^
            --n_ctx %n_ctx% ^
            --t_n_ctx %t_n_ctx%
        wait
done

:: 训练 mvtec 数据集的循环
for i in "%depth%".split() do ^
    for j in "%n_ctx%".split() do ^
        set base_dir=%depth%_%n_ctx%_%t_n_ctx%_multiscale
        set save_dir=..\checkpoints\%base_dir%
        CUDA_VISIBLE_DEVICES=%device% cmd /c call test.py ^
            --dataset mvtec ^
            --data_path "%remote-home%\iot_zhouqihang\data\mvdataset" ^
            --save_path %save_dir%\zero_shot ^
            --checkpoint_path %save_dir%epoch_15.pth ^
            --features_list 6 12 18 24 ^
            --image_size 518 ^
            --depth %depth% ^
            --n_ctx %n_ctx% ^
            --t_n_ctx %t_n_ctx%
        wait
done

:: 将变量转换为字符串输出
echo %LOG%