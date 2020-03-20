# 3D-medical-image-classification
机器学习课程大作业
kaggle competition
https://www.kaggle.com/c/sjtu-m3dv-medical-3d-voxel-classification/

一句话简介：基于3D卷积神经网络的医学图像分类

目标：得到test集的二分类结果，即该肺部结节是否表征某基因
配置：AWS g3.4xlarge


主要使用的代码框架
主要数据预处理方法：mixup、rotation

运行模型：train.py  
数据存放在data中(未上传)   
其中  
函数info中存放info.csv地址  
函数test中存放test.csv地址  
函数nodule_path中存放训练数据的地址  
函数test_nodule_path中存放测试数据的地址

