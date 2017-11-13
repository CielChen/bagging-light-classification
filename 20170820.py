#载入需要的库
import numpy as np  #科学计算库
import pandas as pd  #数据分析
from sklearn import cross_validation  #交叉验证库
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier  #随机森林算法库
from sklearn.grid_search import GridSearchCV
from sklearn import svm
from sklearn.linear_model import SGDClassifier
import matplotlib.pylab as plt
#get_ipython().magic('matplotlib inline')

#step1. 载入训练集，查看数据的类别分布
train = pd.read_csv('F:\Code\\random forest\data\\train.csv')
target='Type'  #Type的值就是二元分类的输出
IDcol='ID'
#train['Type'].value_counts()  #通过jupyter notebook可以看到训练文件中的分类情况

#step2. 选择样本特征和类别输出
x_columns = [x for x in train.columns if x not in [target, IDcol]]  
X = np.array(train[x_columns])  #X：特征
y = np.array(train['Type'])  #y：类别
#X.shape, y.shape   #通过jupyter notebook可以看到特征和类别的维度

# fit the model and get the separating hyperplane
clf = svm.SVC(kernel = 'linear', C = 1.0)
clf.fit(X, y)

w = clf.coef_[0]
a = -w[0] / w[1]  # a可以理解为斜率
xx = np.linspace(-5, 5)
yy = a * xx - clf.intercept_[0] / w[1]  # 二维坐标下的直线方程

# get the separating hyperplane using weighted classes
wclf = svm.SVC(kernel = 'linear', class_weight = {1 : 10})
wclf.fit(X, y)

ww = wclf.coef_[0]
wa = -ww[0] / ww[1]
wyy = wa * xx - wclf.intercept_[0] / ww[1]  # 带权重的直线

# plot separating hyperplanes and samples
h0 = plt.plot(xx, yy, 'k-', label='no weights')
h1= plt.plot(xx, wyy, 'k--', label='with weights')
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.legend()

plt.axis('tight')
plt.show()







# In[ ]:




