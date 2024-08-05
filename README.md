# 这是yolov8_s_mshead的代码储存库
## 注意!代码库并不包含权重文件,请下载[yolov8_s_500epoch](https://download.openmmlab.com/mmyolo/v0/yolov8/yolov8_s_syncbn_fast_8xb16-500e_coco/yolov8_s_syncbn_fast_8xb16-500e_coco_20230117_180101-5aa5f0f1.pth)
## 并保存于"mmrotate_cut/checkpoints/yolov8_s_orin_pretrain"
## 同时将我们传输的权重文件保存于mmrotate_cut/checkpoints/yolov8_s_mshead文件夹下

## 虚拟环境的配置:
    基础环境:linux/windows操作系统 , Cuda == 11.8 , 安装anaconda
    conda create -n mmrotate_rc1 python=3.8.19 -y
    conda activate mmrotate-1.0.0rc1
    pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118
    pip install -U openmim
    mim install mmengine
    mim install "mmcv==2.0.0”
    mim install “mmdet==3.0.0”
## 注意,安装mmrotate库前需要先通过git拉取我们的代码
    git clone https://github.com/231055558/mmrotate_cut.git
    cd mmrotate_cut && pip install .

# 利用我们的模型训练:
## 在控制台输入如下指令(激活您的虚拟环境并进入模型文件目录)
    python configs/yolov8_s_mshead/train/s2anet-le90_yolo_simple_ms_rr_adamW_1x_dota.py --work-dir <your_work_path_file>
其中<your_work_path_file>是你指定的训练结果保存文件夹位置\
训练的输出结果会有.log格式的日志文件和每个epoch的模型权重参数文件,以及最后一次保存的模型权重
# 利用我们的模型权重进行测试
## 在控制台输入如下指令(激活您的虚拟环境并进入模型文件目录)
    todo
