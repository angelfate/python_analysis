#!/usr/bin/env Python3
# -*- coding: utf-8 -*-
# @Software: PyCharm
# @virtualenv：workon
# @contact: Kmeans聚类算法，数据集是Iris(鸢尾花的数据集)，分类数k是3，数据维数是4。
# @Desc：Code descripton
__author__ = '未昔/AngelFate'
__date__ = '2019/8/17 21:00'
import pandas as pd
import numpy as np
import matplotlib.pylab as plt

"""
K-means聚类算法是典型的基于距离的非层次聚类算法，在最小化误差函数的基础上将数据划分为预定的K个类，使得K个类达到类内数据距离之和最小而类间距离之和最大。它是无监督学习算法，采用距离作为相似性的度量指标，即认为两个对象距离越近，其相似性就越大。
1、数据类型与相似性度量
(1)连续属性和离散属性数据
对于连续属性，要依次对每个属性的属性值进行零-均值化处理；对于离散属性，要依次对每个属性的属性值进行数值化处理。然后通过计算距离来度量相似性，K-means聚类算法中一般需要计算样本间的距离，样本和簇的距离，簇和簇的距离。其中，样本间的距离通常用欧式距离(欧几里得距离)、曼哈顿距离和闵可夫斯基距离，样本和簇的距离可以用样本到簇中心的距离代替，簇和簇距离可以用簇中心到簇中心的距离代替。
"""

data = pd.read_table('Iris_data.txt', sep=' ', encoding='utf8',index_col=False,names=['a','b','c','d'])
x = data[['a', 'b', 'c', 'd']].values
print('x:\n',x)

from sklearn.cluster import KMeans

k = 4
iteration = 500

model = KMeans(n_clusters=k, n_jobs=1, max_iter=iteration)
y = model.fit_predict(x)
label_pred = model.labels_
centroids = model.cluster_centers_
inertia = model.inertia_

print('y:\n',y)
print('聚类标签:\n',label_pred)
print('聚类中心:\n',centroids)
print('聚类准则的总和:\n',inertia)

print('----分类结果----:')
result = list(zip(y, x))
for i in result:
    print(i)


r1 = pd.Series(model.labels_).value_counts()
print('r1:\n',r1)

r2 = pd.DataFrame(model.cluster_centers_)
print('r2: \n', r2)

r = pd.concat([r2, r1], axis=1)
r.columns = data.columns.tolist() + ['类别数目']
print('r: \n', r)

file = open('result.txt','w',encoding='utf8')
file.write(str(r1)+'\n\n'+str(r2)+'\n\n'+str(r))
file.close()

