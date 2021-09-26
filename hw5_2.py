# -*- coding: utf-8 -*-
"""HW5-2.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1lcWQfS862UoA3FalfCJLjbJ51qJt7cur
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from mpl_toolkits import mplot3d
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
from sklearn import metrics

col_names = ['Area','Perimeter','Compactness','Length of kernel','Width of kernel','Asymmetry coefficient','Length of kernel groove','target']
data = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/00236/seeds_dataset.txt", header=None, names=col_names,sep='\s+', engine='python')
data.head()

features = data.iloc[:, 0:7]
target = data.iloc[:, -1]

model = KMeans()
visualizer = KElbowVisualizer(model, k=(1,10))

visualizer.fit(features)
visualizer.poof()

kmeans = KMeans(n_clusters=3)
kmeans.fit(features)

kmeans.fit(data.drop('target',axis=1))

centers = kmeans.cluster_centers_
centers

data['klabels'] = kmeans.labels_
data.head()

f, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, sharey=True,figsize = (12,8) )

ax1.set_title('K Means (K = 3)')
ax1.scatter(x = data['Area'], y = data['Asymmetry coefficient'], 
            c = data['klabels'], cmap='rainbow')
ax1.scatter(x=centers[:, 0], y=centers[:, 5],
            c='black',s=300, alpha=0.5);


ax2.set_title("Original")
ax2.scatter(x = data['Area'], y = data['Asymmetry coefficient'], 
            c = data['target'], cmap='rainbow')

fig = plt.figure()
ax = plt.axes(projection='3d')

z1 = data['klabels']
x1 = data['Area']
y1 = data['Asymmetry coefficient']
z2 = data['target']
x2 = data['Area']
y2 = data['Asymmetry coefficient']

ax.scatter(x1, y1, z1, c=z1, cmap='Greens', marker='*', label='K Means')
ax.scatter(x2, y2, z2, c=z2, cmap='Blues', marker='.', label='Original')

ax.legend()

plt.show()