'''
Date: 11/13/2017
Author: CIEL
Function: 批量处理，用基于GBDT（梯度提升树）的分类器标注测试图片中的光源
'''

from time import time
import os
import sys
import re
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import cross_validation, metrics
from sklearn.grid_search import GridSearchCV
import matplotlib.pylab as plt
from sklearn.cross_validation import train_test_split
from sklearn.externals import joblib
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# 生成数据字典
# 输入：dataFile.csv文件
# 输出：一个字典，同sklearn自带的iris数据集
def generateDataDict(path, dataFile):
    # dataFile += path +
    # dataFile = open(path + "\\input\%s" % filename, "r")
    # print (dataFile)
    # if os.path.exists(dataFile):
    file = path + "\\testFeature\%s" %dataFile
    print (file)
    if os.path.exists(file):
        # 生成title
        listTitle = []
        listTitle.append('class')  # 类别
        # 特征
        # featureNum = 340  # 20171023BalanceTrain.csv，共有340维特征（超像素）
        featureNum = 342  # 20170925BalanceDeleteZeroTrain.csv，共有342维特征（像素位置+超像素）
        for i in range(1, featureNum + 1):
            listTitle.append('feature' + str(i))

        # read the csv file
        # pd.read_csv: 封装在DataFrame数据结构中
        dataMatrix = np.array(pd.read_csv(file, header=None, skiprows=1, names=listTitle))  # skiprows=1跳过第一行
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


def floatRange(start,stop,nums):
    ''' Computes a range of floating value.

        Input:
            start (float)  : Start value.
            end   (float)  : End value
            nums (integer): Number of values

        Output:
            A list of floats

        Example:
             print floatRange(0.4, 1, 13)
            [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
    '''
    return [start+float(i)*(stop-start)/(float(nums)-1) for i in range(nums)]


# 用测试集验证模型，单张图片
from matplotlib.font_manager import FontProperties
def predictTest(X_test, y_test, path, dataFile):
    # 加载模型
    print ("load GBDT model ...")
    model = joblib.load('clfGBDT.model')

    # 用测试集验证最优模型
    t0 = time()
    print("Predicting label on the test set")
    y_pre = model.predict(X_test)

    # 创建output file（label=0/1 feature1 feature2）
    portion = os.path.splitext(dataFile)  #将文件的名称和扩展名分离, portion[0]=图片名，portion[1]='.csv'
    outputFileName = path + "\\testResult\%s" % portion[0] + ".txt"
    print (outputFileName)
    print("create modelResult.txt")
    # outputFileName = "modelResult.txt"
    outputFile = open(outputFileName, 'w')
    line = len(y_pre)   # txt的行
    for i in range(0, line):
        outputFile.write(str(y_pre[i]))
        outputFile.write('\t')
        outputFile.write(str(X_test[i][0]))
        outputFile.write('\t')
        outputFile.write(str(X_test[i][1]))
        outputFile.write('\n')
    outputFile.close()
    print("modelResult.txt finished ...")

    y_preprob = model.predict_proba(X_test)[:, 1]
    print ('Accuaracy: %.4g' % metrics.accuracy_score(y_test, y_pre))
    print ('AUC Score (Test): %f' % metrics.roc_auc_score(y_test, y_preprob))
    print("done in %0.3fs" % (time() - t0))

    # # classification_report
    # # 输入：测试集真实的结果和预测的结果
    # # 返回：每个类别的准确率召回率F值以及宏平均值。
    # print("classification report:")
    # print(classification_report(y_test, y_pre))
    #
    # # 混淆矩阵
    # # 输出：
    # # 第0行第0列的数表示y_true中值为0，y_pred中值也为0的个数；第0行第1列的数表示y_true中值为0，y_pred中值为1的个数
    # # 第1行第0列的数表示y_true中值为1，y_pred中值也为0的个数；第1行第1列的数表示y_true中值为1，y_pred中值为1的个数
    # matrix = confusion_matrix(y_test, y_pre)
    # print("confusion matrix:")
    # print(matrix)
    # # 画出混淆矩阵
    # font = FontProperties(fname=r"c:\windows\fonts\msyh.ttc", size=10)   # 字体
    # plt.matshow(matrix)
    # plt.colorbar()
    # plt.xlabel('预测类型', fontproperties=font)
    # plt.ylabel('实际类型', fontproperties=font)
    # labels = ['0', '1']
    # plt.xticks(np.arange(matrix.shape[1]), labels)
    # plt.yticks(np.arange(matrix.shape[1]), labels)
    # plt.show()

# 生成result.txt，用于标注出图片中的光源（单张图片）
def lightSource(path, dataFile):
    portion = os.path.splitext(dataFile)  # 将文件的名称和扩展名分离, portion[0]=图片名，portion[1]='.csv'
    inputFileName = path + "\\testResult\%s" % portion[0] + ".txt"
    print("create lightsOnly.txt")
    ouputFileName = path + "\\testLabel\%s" % portion[0] + ".txt"
    print (ouputFileName)
    outputFile = open(ouputFileName, 'w')
    modelResult = open(inputFileName, "r")
    # 逐行读取
    fileEnd = 0  # fileEnd：读到txt文件结尾的标志。0表示没有到结尾，1到了结尾
    rowNum = 0  # rowNum：txt的行数
    while not fileEnd:
        lineContent = modelResult.readline()  # lineContent：每行内容
        if (lineContent != ''):
            rowNum = rowNum + 1
            # 默认分隔符为空格（不管有几个空格），进行分割字符串lineContent，存入splitLine中
            splitLine = lineContent.split()
            colNum = len(splitLine)  # colNum=列数，下标0~n-1
            # print(colNum)
            # 修改格式为:
            # 图片名.jpg，feature1 feature2
            if(splitLine[0] == '1.0'):
                outputFile.write(portion[0] + '.jpg')  # 图片名
                outputFile.write(' ')
                outputFile.write(splitLine[1])    # 超像素中心
                outputFile.write(' ')
                outputFile.write(splitLine[2])
                outputFile.write('\n')
        else:
            fileEnd = 1
    outputFile.close()
    print("lightOnly.txt finished ...")


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
    # ---------- 列出..\testFeature\文件夹下中的所有文件名，返回的是一个列表 ----------
    path = sys.path[0]  # 获取脚本(.py)路径
    print (path)
    filesDir = os.listdir(path + "\\testFeature")  # ..\testFeature\文件夹
    print (filesDir)

    # ---------- 遍历整个文件夹 ----------
    for filename in filesDir:  # 为每一个测试图片标注光源
        print (filename)
        # ------------------- step 1. 生成数据字典 --------------------------
        # 载入测试数据
        dataFile = filename  # 注意：该csv文件没有'ID'列
        portion = os.path.splitext(filename)  # 将文件的名称和扩展名分离, portion[0]='文件名'，portion[1]='.txt"
        dataFileName = portion[0]
        print (dataFileName)
        print('生成数据字典')
        # 生成训练集数据字典
        dataDict = generateDataDict(path, dataFile)
        # print(dataDict)

        # ------------------- step 2. 提取特征和类别 -------------------
        X = dataDict["feature"]
        print(X)
        y = dataDict["target"]
        print (y)

        # # ------------------- step 2. 原始数据可视化 -------------------
        # print ("原始图像可视化")
        # visualizationDataIn2D(X, y)  # 2维可视化
        # visualizationDataIn3D(X, y)  # 3维可视化

        # -------------------------------- step 3. 输出测试集的label --------------------------------
        predictTest(X, y, path, dataFile)

        # -------------------------------- step 7. 选出 modelResult.txt 中 label =1 的行 --------------------------------
        lightSource(path, dataFile)


if __name__ == '__main__':
    start = time()
    print('program starting ...')

    main()
    end = time()

    print('program ends')
    print('用时 %.5f seconds.' % (end - start))