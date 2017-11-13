'''
Date: 10/25/2017
Author: CIEL
Function: 重新梳理分类器
'''

from time import time
import os
import numpy as np  #科学计算库
import pandas as pd  #数据分析
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.externals import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

# 生成数据字典
# 输入：dataFile.csv文件
# 输出：一个字典，同sklearn自带的iris数据集
def generateDataDict(dataFile):
    if os.path.exists(dataFile):
        # 生成title
        listTitle = []
        listTitle.append('class')  # 类别
        # 特征
        # featureNum = 340  # 20171023BalanceTrain.csv，共有340维特征（超像素）
        featureNum = 342  # 20170925BalanceTrain.csv，共有342维特征（像素位置+超像素）
        for i in range(1, featureNum + 1):
            listTitle.append('feature' + str(i))

        # read the csv file
        # pd.read_csv: 封装在DataFrame数据结构中
        dataMatrix = np.array(pd.read_csv(dataFile, header=None, skiprows=1, names=listTitle))  # skiprows=1跳过第一行
        # print (dataMatrix)

        # 获取样本总数（rowNum）和每个样本的维度（colNum: 类别+特征(featureNum)，共featureNum+1维）
        rowNum, colNum = dataMatrix.shape[0], dataMatrix.shape[1]
        # print(rowNum)
        # print(colNum)

        dataFeature = []  # 样本特征
        dataClass = []  # 样本类别
        for i in range(0, rowNum):  # 遍历全部样本
            # tolist()：转为list类型
            tempList = dataMatrix[i, :].tolist()  # 第i个样本
            dataClass.append(tempList[0])  # 类别
            dataFeature.append(tempList[1:])  # 特征
        featureArray = np.array(dataFeature)  # 二维矩阵，一行是一个样本，行数=样本总数，列数=样本特征数
        # print(featureArray)
        # print(featureArray.shape)
        classArray = np.array(dataClass)  # 一维列向量，每个元素是对应每个样本的所属类别
        # print (classArray)
        # print(classArray.shape)

        dataDict = {}  # 创建字典，用于存储生成的特征和类别
        dataDict['feature'] = np.array(featureArray)
        dataDict['target'] = np.array(classArray)
        # print ("dataDict:")
        # print(dataDict)

        return dataDict
    else:  # 读取csv文件失败
        print('No such file or directory!')

# 2维数据可视化
import matplotlib.pyplot as plt
def visualizationDataIn2D(feature, target):
    # 以feature0和feature1为x轴和y轴
    featureX = 0
    featureY = 1
    x_min, x_max = feature[:, featureX].min() - .5, feature[:, featureX].max() + .5
    y_min, y_max = feature[:, featureY].min() - .5, feature[:, featureY].max() + .5

    #----------- 2维图像 ------------
    figure = plt.figure(2, figsize=(8, 6))   # 支持一种MATLAB式的编号架构，如plt.figure(2)。通过plt.gct()可得到当前figure的引用
                                    # figsize：确保图片保存到磁盘时具有一定的大小和纵横比
    plt.clf()  # clear the entire current figure

    # Plot the training points
    # 设置标题
    ax1 = figure.add_subplot(111)
    # 设置标题
    ax1.set_title('Scatter Plot')
    # 绘制散点图scatter(x, y, ···)
    # x,y是长度相同的数组序列；c是颜色；cmap是colorMap
    plt.scatter(feature[:, 0], feature[:, 1], c=target, cmap=plt.cm.Paired)
    # 设置坐标轴的名字
    plt.xlabel('feature' + str(featureX))
    plt.ylabel('feature' + str(featureY))
    # 设置坐标轴的范围
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    # 自定义坐标
    plt.xticks(())
    plt.yticks(())
    # pycharm显示图像必须有此语句
    plt.show()

# 3维可视化
from mpl_toolkits.mplot3d import Axes3D  # 绘制三维视图
from sklearn.decomposition import PCA
def visualizationDataIn3D(feature, target):
    # PCA：主成分分析，用于降维
    # n_components：保留下来的特征个数n。n_components='mle'，将自动选取特征个数n，使得满足所要求的方差百分比。
    # fit_transform(X)：用X来训练PCA模型，同时返回降维后的数据。newX=pca.fit_transform(X)，newX就是降维后的数据。
    X_reduced = PCA(n_components=3).fit_transform(feature)  # 使用PCA演算法将特征维度降低

    # 利用 scatter以三個特徵資料數值當成座標繪入空間，並以三種iris之數值 Y，來指定資料點的顏色。
    # To getter a better understanding of interaction of the dimensions
    # plot the first three PCA dimensions
    fig = plt.figure(1, figsize=(8, 6))
    ax = Axes3D(fig, elev=-150, azim=110)  # 绘制三维视图
    ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=target,
               cmap=plt.cm.Paired)
    ax.set_title("First three PCA directions")
    ax.set_xlabel("1st eigenvector")
    ax.w_xaxis.set_ticklabels([])
    ax.set_ylabel("2nd eigenvector")
    ax.w_yaxis.set_ticklabels([])
    ax.set_zlabel("3rd eigenvector")
    ax.w_zaxis.set_ticklabels([])

    plt.show()

# PCA特征降维
# 输入：指定降维后的主成分方差和比例
def PCAreduceFeatureNum(percentage):
    print ("特征降维")
    '''
    # 先不降维，只对数据进行投影，看看投影后的340个维度的方差分布
    featureNum = 340   #降维后的维度
    pca = PCA(n_components=featureNum)
    '''
    pca = PCA(n_components=percentage)

    '''
    
    print ("PCA variance")
    print(pca.explained_variance_ratio_)  # explained_variance：代表降维后的各主成分的方差值。方差值越大，则说明越是重要的主成分
    print(pca.explained_variance_)  # explained_variance_ratio_：代表降维后的各主成分的方差值占总方差值的比例，这个比例越大，则越是重要的主成分。
    componentNum = pca.n_components_  # 降至componentNum维
    print(componentNum)
    '''
    return pca


# 用GridSearchCV寻找最优参数，训练SVC模型
# 输入：降维后的训练集特征，训练集类别；降维后的测试集特征，测试集类别
def trainModel(X_train_reduced, y_train, X_test_reduced, y_test):
    print("Fitting the classifier to the training set")
    t0 = time()

    # 参数范围
    param_grid_C = [2**c for c in range(0, 20)]
    param_grid_gamma = [2** c for c in range(-5, 10)]
    param_grid = [{'C': param_grid_C, 'gamma': param_grid_gamma, 'kernel': ['rbf']}, # rbf核需调优C和gamma
                  {'C': param_grid_C, 'kernel': ['linear']}  # 线性核需调优C
                  ]

    # 调优
    # scoring：准确度评价标准
    # cv=n：交叉验证cv参数，n折
    # 这里的GridSearchCV针对每个参数进行了cv次交叉验证，并且一共对cv*param_grid个参数进行相同过程的交叉验证
    # verbose：日志冗长度，int：冗长度，0：不输出训练过程，1：偶尔输出，>1：对每个子模型都输出
    # n_jobs=-1时，可以实现并行计算
    clf = GridSearchCV(SVC(), param_grid, scoring='roc_auc', cv=3, n_jobs=-1, verbose=2)
    clf = clf.fit(X_train_reduced, y_train)

    # 分析结果自动保存
    cv_result = pd.DataFrame.from_dict(clf.cv_results_)
    with open('cv_result.csv', 'w') as f:
        cv_result.to_csv(f)

    print("done in %0.3fs" % (time() - t0))
    print("Best estimator found by grid search:")

    # 最优参数
    best_parameters = clf.best_estimator_.get_params()

    # 最优模型
    model = clf.best_estimator_
    print(clf.best_estimator_)

    # 保存模型
    print ("saving SVC model ...")
    joblib.dump(model, 'clfSVC.model')

    # 用测试集验证最优模型
    print("Predicting label on the test set")
    t0 = time()
    y_pred = model.predict(X_test_reduced)
    print("done in %0.3fs" % (time() - t0))

    # classification_report
    # 输入：测试集真实的结果和预测的结果
    # 返回：每个类别的准确率召回率F值以及宏平均值。
    print(classification_report(y_test, y_pred))

    # confusion_matrix：评估分类器准确性
    # 输出：
    # 第0行第0列的数表示y_true中值为0，y_pred中值也为0的个数；第0行第1列的数表示y_true中值为0，y_pred中值为1的个数
    # 第1行第0列的数表示y_true中值为1，y_pred中值也为0的个数；第1行第1列的数表示y_true中值为1，y_pred中值为1的个数
    print(confusion_matrix(y_test, y_pred))

# 用测试集验证模型
from matplotlib.font_manager import FontProperties
def predictTest(X_test_reduced, y_test):
    # 加载模型
    print ("load SVC model ...")
    model = joblib.load('clfSVC.model')

    # 用测试集验证最优模型
    print("Predicting label on the test set")
    t0 = time()
    y_pred = model.predict(X_test_reduced)
    print("done in %0.3fs" % (time() - t0))

    # classification_report
    # 输入：测试集真实的结果和预测的结果
    # 返回：每个类别的准确率召回率F值以及宏平均值。
    print("classification report:")
    print(classification_report(y_test, y_pred))

    # 混淆矩阵
    # 输出：
    # 第0行第0列的数表示y_true中值为0，y_pred中值也为0的个数；第0行第1列的数表示y_true中值为0，y_pred中值为1的个数
    # 第1行第0列的数表示y_true中值为1，y_pred中值也为0的个数；第1行第1列的数表示y_true中值为1，y_pred中值为1的个数
    matrix = confusion_matrix(y_test, y_pred)
    print("confusion matrix:")
    print(matrix)
    # 画出混淆矩阵
    font = FontProperties(fname=r"c:\windows\fonts\msyh.ttc", size=10)   # 字体
    plt.matshow(matrix)
    plt.colorbar()
    plt.xlabel('预测类型', fontproperties=font)
    plt.ylabel('实际类型', fontproperties=font)
    labels = ['0', '1']
    plt.xticks(np.arange(matrix.shape[1]), labels)
    plt.yticks(np.arange(matrix.shape[1]), labels)
    plt.show()


from matplotlib.colors import ListedColormap
# 画出分界区域
def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                            np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, slpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot all samples
    X_test, y_test = X[test_idx, :], y[test_idx]
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=cmap(idx),
                    marker=markers[idx], label=cl)

    # highlight test samples
    if test_idx:
        X_test, y_test = X[test_idx, :], y[test_idx]
        plt.scatter(X_test[:, 0], X_test[:, 1], c='',
                    alpha=1.0, linewidth=1, marker='o',
                    s=55, label='test set')

# 主函数
def main():
    #------------------- step 1. 生成数据字典 --------------------------
    # 载入训练数据
    # dataFile = '20171023BalanceTrain.csv'  # 注意：该csv文件没有'ID'列
    dataFile = '20170925BalanceTrain.csv'  # 注意：该csv文件没有'ID'列
    print('生成数据字典')
    # 生成训练集数据字典
    dataDict = generateDataDict(dataFile)
    # print(dataDict)

    # ------------------- 提取特征和类别 -------------------
    X = dataDict["feature"]
    # print(X)
    y = dataDict["target"]
    # print (y)

    # 特征工程参考：http://www.cnblogs.com/jasonfreak/p/5448385.html
    # -------------------------------- 无量纲化：使不同规格的数据转换到同一规格 --------------------------------
    print ("无量纲化")
    # 第一种：区间缩放
    # X = MinMaxScaler().fit_transform(X)   # 利用两个最值进行区间缩放

    # 第二种：标准化
    # 标准化参考：http://blog.csdn.net/pipisorry/article/details/52247679
    X = StandardScaler().fit_transform(X)  # 标准化，转化为正态分布
    # print(X)

    #------------------- step 2. 数据集分割：训练集 + 测试集 --------------------------
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)   # 随机分成两部分，训练集占75%，测试集占25%
    '''
    print("trainDict")
    print("feature")
    print (X_train)
    print("class")
    print (y_train)
    print("testDict")
    print("feature")
    print (X_test)
    print("class")
    print (y_test)
    '''

    '''
    # 原始数据可视化
    print ("原始图像可视化")
    visualizationDataIn2D(X, y)   # 2维可视化
    visualizationDataIn3D(X, y)   # 3维可视化
    '''


    # -------------------------------- 特征降维 --------------------------------
    print ("特征降维")
    '''
    # 先不降维，只对数据进行投影，看看投影后的340个维度的方差分布
    featureNum = 340   #降维后的维度
    pca = PCA(n_components=featureNum)
    '''
    percentage = 0.99  # 指定降维后的主成分方差和比例
    pca = PCAreduceFeatureNum(percentage)

    X_train_reduced = pca.fit_transform(X_train)   # 用X_train来训练PCA模型，同时返回降维后的数据
    X_test_reduced = pca.transform(X_test)  # 用相同的PCA模型对训练集降维
    print('训练集数据的原始维度是：{}'.format(X_train.shape))
    print('PCA降维后训练集数据是：{}'.format(X_train_reduced.shape))


    '''
    # 降维数据的可视化
    print ("降维数据的可视化")
    visualizationDataIn2D(X_train_reduced, y_train)  # 训练集的2维可视化
    '''

    '''
    classifier = LogisticRegression()
    accuracies = cross_val_score(classifier, X_train_reduced, y_train)

    print('交叉验证准确率是：{}\n{}'.format(np.mean(accuracies), accuracies))
    classifier.fit(X_train_reduced, y_train)
    predictions = classifier.predict(X_test_reduced)
    print(classification_report(y_test, predictions))
    '''

    # 训练模型
    trainModel(X_train_reduced, y_train, X_test_reduced, y_test)

    # 验证模型
    predictTest(X_test_reduced, y_test)

    # --------------------------- 画出测试集的分界区域 ---------------------------
    # 加载模型
    print ("load SVC model ...")
    model = joblib.load('clfSVC.model')

    #X_combined = np.vstack((X_train_reduced[:, 1], X_test_reduced[:, 1]))  # 将X_train_reduced和X_test_reduced按行合并为X_combined
    #y_combined = np.hstack((y_train, y_test))  # 将y_train和y_test按列合并为y_combined

    h = 0.02
    X_test_seperation = X_test_reduced
    y_test_seperation = y_test
    x_min, x_max = X_test_seperation[:, 0].min() - .5, X_test_seperation[:, 0].max() + .5
    y_min, y_max = X_test_seperation[:, 1].min() - .5, X_test_seperation[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # 画分界区
    # plot_decision_regions(X_combined, y_combined, model)
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
    plt.scatter(X_test_reduced[:, 0], X_test_reduced[:, 1], c=y_test, cmap=plt.cm.Paired)
    plt.xlabel('sepal length')
    plt.ylabel('sepal width')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())
    plt.title('SVC classification')
    plt.show()

if __name__ == '__main__':
    start = time()
    print('program starting ...')

    main()
    end = time()

    print('program ends')
    print('用时 %.5f seconds.' % (end - start))