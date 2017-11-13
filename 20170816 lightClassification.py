
# coding: utf-8

# In[ ]:


# %load lightCls-RF.py
'''
################## Function: 基于随机森林的光源分类器 ##################
通过随机抽样的方式从数据表中生成多张抽样的数据表，对每个抽样的数据表生成一棵决策树。
将多颗决策树组成一个随机森林。
当有一条新的数据产生时，让森林里的每一颗决策树分别进行判断，
以投票最多的结果作为最终的判断结果。
################## Author: Ciel ##################
################## Date: 08/10/2017 ##################
'''

#random forest调参参考
#http://www.cnblogs.com/pinard/p/6160412.html
#http://bluewhale.cc/2016-12-23/use-python-to-train-random-forest-model-to-identify-suspicious-traffic.html

#载入需要的库
import numpy as np  #科学计算库
import pandas as pd  #数据分析
from sklearn import cross_validation  #交叉验证库
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier  #随机森林算法库
from sklearn.grid_search import GridSearchCV
import matplotlib.pylab as plt
get_ipython().magic('matplotlib inline')

#step1. 载入训练集，查看数据的类别分布
train = pd.read_csv('F:\Code\\random forest\data\\train.csv')
target='Type'  #Type的值就是二元分类的输出
IDcol='ID'
train['Type'].value_counts()  #通过jupyter notebook可以看到训练文件中的分类情况

#step2. 选择样本特征和类别输出
x_columns = [x for x in train.columns if x not in [target, IDcol]]  
X = np.array(train[x_columns])  #X：特征
y = np.array(train['Type'])  #y：类别
X.shape, y.shape   #通过jupyter notebook可以看到特征和类别的维度

#step3. 将训练集分割为训练样本和验证样本
#采用随机的方式将数据表分割为训练集和测试集，其中60%的训练集数据用来训练模型，40%的测试集数据用来检验模型准确率。
X_train, X_validation, y_train, y_validation=cross_validation.train_test_split(X, y, test_size=0.4, random_state=0)
X_train.shape, y_train.shape   #通过jupyter notebook可以看到训练数据的维度
X_validation.shape, y_validation.shape   #通过jupyter notebook可以看到验证数据的维度

#step4. 建立随机森林模型，并训练
#使用训练数据对模型进行训练，并使用验证数据对模型的训练结果进行评估。
clf = RandomForestClassifier()  #建立模型
clf = clf.fit(X_train, y_train)  #训练模型
clf.score(X_validation, y_validation)  #验证模型









# In[10]:


# %load lightCls-RF.py
'''
################## Function: 基于随机森林的光源分类器 ##################
通过随机抽样的方式从数据表中生成多张抽样的数据表，对每个抽样的数据表生成一棵决策树。
将多颗决策树组成一个随机森林。
当有一条新的数据产生时，让森林里的每一颗决策树分别进行判断，
以投票最多的结果作为最终的判断结果。
################## Author: Ciel ##################
################## Date: 08/10/2017 ##################
'''

#random forest调参参考
#http://www.cnblogs.com/pinard/p/6160412.html
#http://bluewhale.cc/2016-12-23/use-python-to-train-random-forest-model-to-identify-suspicious-traffic.html

#载入需要的库
import numpy as np  #科学计算库
import pandas as pd  #数据分析
from sklearn import cross_validation  #交叉验证库
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier  #随机森林算法库
from sklearn.grid_search import GridSearchCV
import matplotlib.pylab as plt
get_ipython().magic('matplotlib inline')

#step1. 载入训练集，查看数据的类别分布
train = pd.read_csv('F:\Code\\random forest\data\\train.csv')
target='Type'  #Type的值就是二元分类的输出
IDcol='ID'
train['Type'].value_counts()  #通过jupyter notebook可以看到训练文件中的分类情况

#step2. 选择样本特征和类别输出
x_columns = [x for x in train.columns if x not in [target, IDcol]]  
X = np.array(train[x_columns])  #X：特征
y = np.array(train['Type'])  #y：类别
X.shape, y.shape   #通过jupyter notebook可以看到特征和类别的维度

#step3. 将训练集分割为训练样本和验证样本
#采用随机的方式将数据表分割为训练集和测试集，其中60%的训练集数据用来训练模型，40%的测试集数据用来检验模型准确率。
X_train, X_validation, y_train, y_validation=cross_validation.train_test_split(X, y, test_size=0.4, random_state=0)
X_train.shape, y_train.shape   #通过jupyter notebook可以看到训练数据的维度
X_validation.shape, y_validation.shape   #通过jupyter notebook可以看到验证数据的维度


#step4. 建立随机森林模型，并训练
#使用训练数据对模型进行训练，并使用验证数据对模型的训练结果进行评估。
clf = RandomForestClassifier()  #建立模型
clf = clf.fit(X_train, y_train)  #训练模型
clf.score(X_validation, y_validation)  #验证模型


# In[27]:


#载入需要的库
import numpy as np  #科学计算库
import pandas as pd  #数据分析
from sklearn import cross_validation  #交叉验证库
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier  #随机森林算法库
from sklearn.grid_search import GridSearchCV
import matplotlib.pylab as plt
get_ipython().magic('matplotlib inline')

#step1. 载入训练集，查看数据的类别分布
train = pd.read_csv('F:\Code\\random forest\data\\train.csv')
target='Type'  #Type的值就是二元分类的输出
IDcol='ID'
train['Type'].value_counts()  #通过jupyter notebook可以看到训练文件中的分类情况

#step2. 选择样本特征和类别输出
x_columns = [x for x in train.columns if x not in [target, IDcol]]  
feature = np.array(train[x_columns])  #feature：特征
label = np.array(train['Type'])  #label：类别
feature.shape, label.shape   #通过jupyter notebook可以看到特征和类别的维度

#step3. 将训练集分成两部分：训练数据和验证数据
#采用“自然（分层）分布”:将数据集划分成k份，不同点在于，划分的k份中，每一份内各个类别数据的比例和原始数据集中各个类别的比例相同
skf = cross_validation.StratifiedKFold(y, n_folds = 2)
len(skf)
for train_index, test_index in skf:
    print("TRAIN:", train_index, "TEST:", test_index)
    feature_train, feature_test = feature[train_index], feature[test_index]
    label_train, label_test = label[train_index], label[test_index]
feature_train.shape, feature_test.shape   #通过jupyter notebook可以看到训练数据的维度    


# In[26]:


#载入需要的库
import numpy as np  #科学计算库
import pandas as pd  #数据分析
from sklearn import cross_validation  #交叉验证库
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier  #随机森林算法库
from sklearn.grid_search import GridSearchCV
import matplotlib.pylab as plt
get_ipython().magic('matplotlib inline')

#step1. 载入训练集，查看数据的类别分布
train = pd.read_csv('F:\Code\\random forest\data\\train.csv')
target='Type'  #Type的值就是二元分类的输出
IDcol='ID'
train['Type'].value_counts()  #通过jupyter notebook可以看到训练文件中的分类情况

#step2. 选择样本特征和类别输出
x_columns = [x for x in train.columns if x not in [target, IDcol]]  
X = np.array(train[x_columns])  #X：特征
y = np.array(train['Type'])  #y：类别
X.shape, y.shape   #通过jupyter notebook可以看到特征和类别的维度

#step3. 将训练集分成两部分：训练数据和验证数据
#采用“自然（分层）分布”:将数据集划分成k份，不同点在于，划分的k份中，每一份内各个类别数据的比例和原始数据集中各个类别的比例相同
skf = cross_validation.StratifiedKFold(y, n_folds = 10)
len(skf)  #显示：k=几折
for train_index, test_index in skf:
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
X_train.shape, X_test.shape   #通过jupyter notebook可以看到训练数据的维度，验证数据的维度
y_train.value_counts()  #通过jupyter notebook可以看到训练数据的分布


# In[24]:


#载入需要的库
import numpy as np  #科学计算库
import pandas as pd  #数据分析
from sklearn import cross_validation  #交叉验证库
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier  #随机森林算法库
from sklearn.grid_search import GridSearchCV
import matplotlib.pylab as plt
get_ipython().magic('matplotlib inline')

#step1. 载入训练集，查看数据的类别分布
train = pd.read_csv('F:\Code\\random forest\data\\train.csv')
target='Type'  #Type的值就是二元分类的输出
IDcol='ID'
train['Type'].value_counts()  #通过jupyter notebook可以看到训练文件中的分类情况
#通过程序，可以看到

#自定义抽样函数，


#step2. 选择样本特征和类别输出
x_columns = [x for x in train.columns if x not in [target, IDcol]]  
X = np.array(train[x_columns])  #X：特征
y = np.array(train['Type'])  #y：类别
X.shape, y.shape   #通过jupyter notebook可以看到特征和类别的维度


# In[ ]:


#载入需要的库
import numpy as np  #科学计算库
import pandas as pd  #数据分析
from sklearn import cross_validation  #交叉验证库
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier  #随机森林算法库
from sklearn.grid_search import GridSearchCV
import matplotlib.pylab as plt
get_ipython().magic('matplotlib inline')

fileName = 'F:\Code\\random forest\data\\train.csv'

#step1. 载入训练集，查看数据的类别分布
#train = pd.read_csv('F:\Code\\random forest\data\\train.csv')
train = pd.read_csv(fileName)
target='Type'  #Type的值就是二元分类的输出
IDcol='ID'
train['Type'].value_counts()  #通过jupyter notebook可以看到训练文件中的分类情况
#通过程序，可以看到：y=0，有210826个；y=1，有11412个


#自定义函数：统计csv文件的行数
def getLineCount(csvFile):
    with open(csvFile, "r", encoding = "utf-8") as f


#自定义函数：划分训练数据和验证数据，采用“分层分布”，训练数据占90%，验证数据占10%
def partitionTrainValid(trainFile):
    for i in range(len(trainFile.readlines())):


# In[34]:


#载入需要的库
import numpy as np  #科学计算库
import pandas as pd  #数据分析
from sklearn import cross_validation  #交叉验证库
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier  #随机森林算法库
from sklearn.grid_search import GridSearchCV
import matplotlib.pylab as plt
get_ipython().magic('matplotlib inline')

fileName = 'F:\Code\\random forest\data\\train.csv'

#step1. 载入训练集，查看数据的类别分布
#train = pd.read_csv('F:\Code\\random forest\data\\train.csv')
train = pd.read_csv(fileName)
target='Type'  #Type的值就是二元分类的输出
IDcol='ID'
train['Type'].value_counts()  #通过jupyter notebook可以看到训练文件中的分类情况
#通过程序，可以看到：y=0，有210826个；y=1，有11412个

csvFile = open(fileName, 'r')
lines = csvFile.readlines()
dataSetNum = len(lines)-1  #-1：要把表头减下去，得到数据集的样本个数
print(dataSetNum)  


# In[ ]:




