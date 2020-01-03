# 3D-medical-image-classification
机器学习课程大作业

运行模型：train.py  
数据存放在data中(未上传)  
mylib.dataloader.path_manager 中存放数据的绝对路径(需要修改)  
其中  
函数info中存放info.csv地址  
函数test中存放test.csv地址  
函数nodule_path中存放训练数据的地址  
函数test_nodule_path中存放测试数据的地址  

已训练好的模型存放在result中  
运行test.py即可得到结果(需要修改test和test_nodule_path的地址)  
result中的5_avg.csv存放的是5次分类结果  
其中p1由于操作失误已找不到原模型,p2用了sampleSubmission  
最终结果是5次分类的平均值  

