## 本项目采用Kaggle开源手机价格预测数据集,以下为数据集链接

https://www.kaggle.com/datasets/atefehmirnaseri/cell-phone-price/data

数据集为手机价格范围的四分类任务

本项目展示了四种多重感知机MLP及不同参数初始化方式的搭配演示,分别为

* cell_ph.py:演示了6层MLP加上sigmoid激活,效果最好,效果图为loss_acc1.png
* cell_ph1.py:演示了6层MLP加上层归一化,sigmoid激活,xavier参数初始化,效果图为loss_acc2.png
* cell_ph2.py:演示了6层MLP加上ReLU激活,kaiming参数初始化,效果图为loss_acc3.png
* cell_ph3.py:演示了6层MLP加上Leaky ReLU激活,kaiming参数初始化,效果图为loss_acc4.png