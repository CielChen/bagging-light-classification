#################################################  
# Author : CIEL 
# Date   : 2017-09-27   
# Function : testing clf1.model-clf5.model
#################################################  

#载入需要的库
import numpy as np  #科学计算库
import pandas as pd  #数据分析
import os

from sklearn import cross_validation  #交叉验证库
from sklearn import metrics
from sklearn import datasets  
from sklearn import preprocessing  
from sklearn import neighbors  
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis  
from sklearn.ensemble import RandomForestClassifier  #随机森林算法库
from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import train_test_split  
from sklearn.model_selection import StratifiedKFold  
from sklearn import svm
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.learning_curve import learning_curve
from sklearn.cross_validation import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from time import time  
from sklearn.naive_bayes import MultinomialNB  
from sklearn import tree  
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier 
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import classification_report 
from sklearn.metrics import precision_recall_curve, roc_curve, auc 
from sklearn.externals import joblib
import matplotlib.pylab as plt
from pandas import DataFrame

# 读取原始数据的csv文件，生成训练集和验证集的数据字典
def setDict_train_validation(originDataFile):
    if os.path.exists(originDataFile):
        # 生成title
        listTitle = []
        #listTitle.append('ID')
        listTitle.append('class')
        for i in range(1, 343):
            listTitle.append('feature' + str(i))
            
        # read the csv file
        # pd.read_csv: 封装在DataFrame数据结构中
        dataMatrix = np.array(pd.read_csv(originDataFile, header=None, skiprows=1, names=listTitle))
        
        # 获取样本总数（rowNum）和每个样本的维度（colNum: 类别+特征，共343维）
        rowNum, colNum = dataMatrix.shape[0], dataMatrix.shape[1]
        
        sampleData = [] # 样本特征
        sampleClass = []  # 样本类别
        for i in range(0, rowNum):  # 遍历全部样本
            #tolist()：转为list类型
            tempList=dataMatrix[i,:].tolist()  # 第i个样本
            #tempList = list(ddd)  # 第i个样本
            sampleClass.append(tempList[0])  # 类别
            sampleData.append(tempList[1:])  # 特征
        sampleM = np.array(sampleData)  # 二维矩阵，一行是一个样本，行数=样本总数，列数=样本特征数
        classM = np.array(sampleClass)  # 一维列向量，每个元素是对应每个样本的所属类别
        
        # from sklearn.model_selection import StratifiedKFold
        # 调用StratifiedKFold生成训练集和测试集
        skf = StratifiedKFold(n_splits=10)  #10折，每次分折都保持原数据集每个类别的样本百分比
        # 大括号{ }：代表dict字典数据类型，字典是由键对值组组成。冒号':'分开键和值，逗号','隔开组。
        setDict = {}  # 创建字典，用于存储生成的训练集和测试集
        count = 1
        for trainI, testI, in skf.split(sampleM, classM):  # skf.split：生成训练样本索引（trainI）和测试样本索引(testI)
            #print('trainI')
            #print(trainI.shape)
            #print('testI: %d')
            #print(testI.shape)
            
            # train1 = SVC1_train; test1 = validation_data
            
            trainSTemp = []  # 存储当前循环抽取出的训练样本特征
            trainCTemp = []  # 存储当前循环抽取出的训练样本类别
            testSTemp = []  # 存储当前循环抽取出的测试样本特征
            testCTemp = []  # 存储当前循环抽取出的测试样本类别
            
            # ------------------- 生成训练集 ------------------- 
            # 第i折中训练数据：trainI
            trainIndex = trainI.tolist()  # 训练样本索引
            for t1 in range(0, len(trainIndex)):
                trainNum =  trainIndex[t1]
                trainSTemp.append((sampleM[trainNum, :]).tolist())
                trainCTemp.append(((classM)[trainNum]).tolist())
            # 第i折中测试数据：testI
            testIndex = testI.tolist()  # 测试样本索引
            for t2 in range(0, len(testIndex)):
                testNum =  testIndex[t2]
                testSTemp.append((sampleM[testNum, :]).tolist())
                testCTemp.append(((classM)[testNum]).tolist())    
            
            # 如果i=1，则生成训练数据和验证数据的字典
            if count == 1:  
                # 生成训练数据
                setDict['trainFeature'] =  np.array(trainSTemp)  # 特征
                setDict['trainClass'] = np.array(trainCTemp)  # 类别
                # 生成验证数据
                setDict['validationFeature'] = np.array(testSTemp)  # 特征
                setDict['validationClass'] = np.array(testCTemp)   # 类别
                
                #print(np.array(trainSTemp).shape)
                #print(np.array(trainCTemp).shape)
                #print(np.array(testSTemp).shape)
                #print(np.array(testCTemp).shape)
               
            else:  # i不等于1，退出for循环
                break
                
            count += 1
        #print(setDict)
        return setDict
    else:  #读取csv文件失败
        print('No such file or directory!') 


# 将训练集和验证集的字典分别写入csv文件，生成train.csv和validation.csv
import csv
def setCSV_train_validation(train_validation_dict, trainFile, validationFile):
    # ------------ train.csv ------------
    with open(trainFile, "w", newline="") as train_file:
        csvWriter = csv.writer(train_file)
        # --------- 先写columns_name -----------
        columnsName = []
        columnsName.append('class')
        for i in range(1, 343):
            columnsName.append('feature' + str(i))
        csvWriter.writerow(columnsName)
        # ------------ 再写数据 --------------
        trainFeature = train_validation_dict['trainFeature']
        trainClass = train_validation_dict['trainClass']
        #print(trainClass.shape[0])  # 输出训练数据的样本数
        #print(trainFeature)
        #print(trainClass)
        for i in range(1, trainClass.shape[0] + 1):  # 写train.csv的每行.trainClass.shape[0]为训练数据的样本数
            rowTemp = []
            rowTemp.append(trainClass[i-1])
            for j in range(0, 342):
                rowTemp.append(trainFeature[i-1, j])
            csvWriter.writerow(rowTemp)
    
    # ------------ validation.csv ------------
    with open(validationFile, "w", newline="") as validation_file:
        csvWriter = csv.writer(validation_file)
        # --------- 先写columns_name -----------
        columnsName = []
        columnsName.append('class')
        for i in range(1, 343):
            columnsName.append('feature' + str(i))
        csvWriter.writerow(columnsName)
        # ------------ 再写数据 --------------
        validationFeature = train_validation_dict['validationFeature']
        validationClass = train_validation_dict['validationClass']
        #print(validationClass.shape[0])  # 验证数据的样本数
        #print(validationFeature)
        #print(validationClass)
        for i in range(1, validationClass.shape[0] + 1):  # validation.csv的每行.validationClass.shape[0]为验证数据的样本数
            rowTemp = []
            rowTemp.append(validationClass[i-1])
            for j in range(0, 342):
                rowTemp.append(validationFeature[i-1, j])
            csvWriter.writerow(rowTemp)

# 读取train.csv文件，生成训练数据的字典，里面含k组采样数据
# 函数返回一个字典：键为集合编号（1SVCFeature, 1SVCClass, 2SVCFeature, 2SVCClass）
def loadTrainDataDict(trainFile):
    if os.path.exists(trainFile):
        # 生成title
        listTitle = []
        listTitle.append('class')
        for i in range(1, 343):
            listTitle.append('feature' + str(i))
            
        # read the csv file
        # pd.read_csv: 封装在DataFrame数据结构中
        dataMatrix = np.array(pd.read_csv(trainFile, header=None, skiprows=1, names=listTitle))
        
        # 获取样本总数（rowNum）和每个样本的维度（colNum: 类别+特征，共343维）
        rowNum, colNum = dataMatrix.shape[0], dataMatrix.shape[1]
        
        sampleData = [] # 样本特征
        sampleClass = []  # 样本类别
        for i in range(0, rowNum):  # 遍历全部样本
            #tolist()：转为list类型
            tempList=dataMatrix[i,:].tolist()  # 第i个样本
            sampleClass.append(tempList[0])  # 类别
            sampleData.append(tempList[1:])  # 特征
        sampleM = np.array(sampleData)  # 二维矩阵，一行是一个样本，行数=样本总数，列数=样本特征数
        classM = np.array(sampleClass)  # 一维列向量，每个元素是对应每个样本的所属类别
        
        # from sklearn.model_selection import StratifiedKFold
        # 调用StratifiedKFold生成训练集和测试集
        foldsNum = 5  # 5折，
        skf = StratifiedKFold(n_splits = foldsNum)  #每次分折都保持原数据集每个类别的样本百分比
        # 大括号{ }：代表dict字典数据类型，字典是由键对值组组成。冒号':'分开键和值，逗号','隔开组。
        setTrainDict = {}  # 创建字典，用于存储生成的训练集和测试集
        count = 1
        for trainI, testI, in skf.split(sampleM, classM):  # skf.split：生成训练样本索引（trainI）和测试样本索引(testI)
            #print('trainI')
            #print(trainI.shape)
            #print('testI: %d')
            #print(testI.shape)
            
            # -----------------------------------------------------------------------------------------------
            # train1 + test1 = SVC1_train; train2 + test2 = SVC2_train; ``` ; train5 + test5 = SVC5_train 
            # -----------------------------------------------------------------------------------------------
            trainSTemp = []  # 存储当前循环抽取出的训练样本特征
            trainCTemp = []  # 存储当前循环抽取出的训练样本类别
            testSTemp = []  # 存储当前循环抽取出的测试样本特征
            testCTemp = []  # 存储当前循环抽取出的测试样本类别
            
            # ------------------- 生成第i个SVC模型的训练集 ------------------- 
            # 第i折中训练数据：trainI
            trainIndex = trainI.tolist()  # 训练样本索引
            for t1 in range(0, len(trainIndex)):
                trainNum =  trainIndex[t1]
                trainSTemp.append((sampleM[trainNum, :]).tolist())
                trainCTemp.append(((classM)[trainNum]).tolist())
            # 第i折中测试数据：testI
            testIndex = testI.tolist()  # 测试样本索引
            for t2 in range(0, len(testIndex)):
                testNum =  testIndex[t2]
                trainSTemp.append((sampleM[testNum, :]).tolist())
                trainCTemp.append(((classM)[testNum]).tolist())    
            
            if count in range(1, foldsNum+1): 
                setTrainDict[str(count) + 'SVCFeature'] =  np.array(trainSTemp)
                setTrainDict[str(count) + 'SVCClass'] = np.array(trainCTemp)
                #print(np.array(trainSTemp).shape)
                #print(np.array(trainCTemp).shape)
            
            count += 1
        #print(setTrainDict)
        return setTrainDict
    else:  #读取csv文件失败
        print('No such file or directory!')  
        
        
# 读取validation.csv文件，生成验证数据的字典
# 函数返回一个字典：键为集合编号（validationFeature, validationClass）
def loadValidationDataDict(validationFile):
    if os.path.exists(validationFile):
        # 生成title
        listTitle = []
        listTitle.append('class')
        for i in range(1, 343):
            listTitle.append('feature' + str(i))
            
        # read the csv file
        # pd.read_csv: 封装在DataFrame数据结构中
        dataMatrix = np.array(pd.read_csv(validationFile, header=None, skiprows=1, names=listTitle))
        
        # 获取样本总数（rowNum）和每个样本的维度（colNum: 类别+特征，共343维）
        rowNum, colNum = dataMatrix.shape[0], dataMatrix.shape[1]
        
        sampleData = [] # 样本特征
        sampleClass = []  # 样本类别
        for i in range(0, rowNum):  # 遍历全部样本
            #tolist()：转为list类型
            tempList=dataMatrix[i,:].tolist()  # 第i个样本
            sampleClass.append(tempList[0])  # 类别
            sampleData.append(tempList[1:])  # 特征
        sampleM = np.array(sampleData)  # 二维矩阵，一行是一个样本，行数=样本总数，列数=样本特征数
        classM = np.array(sampleClass)  # 一维列向量，每个元素是对应每个样本的所属类别
        
        setValidationDict = {}  # 创建字典，用于存储生成的特征和类别
        setValidationDict['validationFeature'] =  np.array(sampleM)
        setValidationDict['validationClass'] = np.array(classM)
        
        #print(np.array(sampleM).shape)
        #print(np.array(classM).shape)
        #print(setValidationDict)
        
        return setValidationDict
    else:  #读取csv文件失败
        print('No such file or directory!')  
            
#支持向量机（Support Vector Machine）:SVC  
def SVC():  
    # 将probability设为True,可以计算每个样本到各个类别的概率
    clf = svm.SVC(probability = True)  
    return clf  


# 计算识别率
# modelClass为模型判断的类别，preClass为先验类别
def getRecognitionRate(modelClass, preClass):
    validationNum = len(modelClass)  # 验证集大小
    rightNum =0  # 分类正确的验证样本个数
    # print('验证集大小')
    # print(validationNum)
    for i in range(0, validationNum):
        if modelClass[i] == preClass[i]:
            rightNum += 1
    # print('正确分类的样本数')
    # print(rightNum)
    return float(rightNum)/ float(validationNum)

# 5个SVC的bagging
def bagging_SVC_model(trainDict, validationDict):  # 输入参数为：训练集的字典，验证集的字典
    # 获取第i个SVC分类器  
    clf_SVC = SVC()  
    
    # 获取基分类器个数（即SVC模型个数）
    setNums = int(len(trainDict.keys()) / 2)  # setNums个SVC模型
    #print(setNums)
    
    #SVC_rate = 0.0  # SVC_rate用于将每个SVC模型的所有识别率累加
    SVC_predict_result = []  # SVC_predict_result记录每个SVC模型对验证数据所属类别的判断
    
    # 验证数据
    validationFearuteMatrix = validationDict['validationFeature']  # 特征
    validationClass = validationDict['validationClass']  # 类别

    for i in range (1, setNums+1):   # i个SVC模型
        trainFeatureMatrix = trainDict[str(i) + 'SVCFeature']  # 第i个SVC模型的训练数据的特征
        trainClass = trainDict[str(i) + 'SVCClass']  # 第i个SVC模型的训练数据的类别
        
        start = time()
        
        # ---------- 训练第i个SVC模型 ------------- 
        # SVC.fit(X, y[, sample_weight])：Fit the SVM model according to the given training data
        clf_SVC.fit(trainFeatureMatrix, trainClass)
        
        # 保存模型
        print('saving '+str(i)+' model')
        joblib.dump(clf_SVC, 'clf'+str(i)+'.model')
        
# 加载模型，并预测数据
# dataDict为验证样本的数据字典
def predictClass_new(validationDict):  
    # 验证数据
    validationFeatureMatrix = validationDict['validationFeature'] #特征
    validationClass = validationDict['validationClass'] #类别
    
    SVC_predict_result = []  # SVC_predict_result记录每个SVC模型对验证数据所属类别的判断
    
    # 遍历每个基分类器
    for i in range(1, 6):   # 共5个基分类器
        # 加载第i个基模型
        print('加载第' + str(i) + '个model')
        iSVCModel = joblib.load('clf'+str(i)+'.model')
        # 计算第i个SVC模型对验证数据的类别判断结果
        iSCV_predictClass = iSVCModel.predict(validationFeatureMatrix)  # predict(X): 返回X的分类（二分类，返回0和1）
        SVC_predict_result.append(iSCV_predictClass.tolist())
        
    # ---------------- bagging判断结果：5个SVC投票 ------------------
    baseModelNums = np.array(SVC_predict_result).shape[0]   # baseModelNums: 基分类器的个数（即5个SVC）
    validationSampleNums = np.array(SVC_predict_result).shape[1]   # validationSampleNums: 验证集的样本数
    
    bagging_predict_statistic = []  # bagging全部基分类器的分类结果统计
    # 初始化bagging统计结果
    for i in range(0, validationSampleNums):
        bagging_predict_statistic.append(0)
    
    # 统计每个基分类器的判断结果
    for i in range(1, baseModelNums+1):
        for j in range(0, validationSampleNums):
            if(np.array(SVC_predict_result)[i-1, j] > 0):  # 第i个基分类器分类为1，则结果+1
                bagging_predict_statistic[j] = bagging_predict_statistic[j] + 1
            '''
            else:   #第i个基分类器分类为0，则结果-1
                bagging_predict_statistic[j] = bagging_predict_statistic[j] - 1
            '''
        #print('ith_bagging_predict_result')
        #print(bagging_predict_result) 
    #print('bagging_predict_statistic')
    #print(bagging_predict_statistic)     
    
    # 通过vote，判断bagging的最终结果
    bagging_predict_cls_result = []
    for i in range(0, validationSampleNums):
        if bagging_predict_statistic[i] == 0:
            bagging_predict_cls_result.append(0)
        else:
            bagging_predict_cls_result.append(1)
    print('bagging_predict_cls_result')
    print(bagging_predict_cls_result)     
    
    # ----------------------------- 计算bagging的识别率 ----------------------------- 
    #bagging_rate = 0.0  # bagging的所有识别率
    #bagging_rate = getRecognitionRate(bagging_predict_cls_result, validationClass)  
    #print('\n')
    #print('Bagging recognition rate: ', bagging_rate) 
    
    # ----------------------------- 模型表现 ----------------------------- 
    # classification_report()：打印分类结果报告
    print('\n')
    print('bagging classification report')
    print(classification_report(validationClass, bagging_predict_cls_result, target_names = ['neg', 'pos']))
    bagging_rate = getRecognitionRate(bagging_predict_cls_result, validationClass) # 验证集的识别率
    print("Bagging average precision:", bagging_rate)
        

def main():

    '''
    # --------------------- step1. 生成训练集的csv文件(train.csv)和验证集的csv文件(validation.csv) ---------------------
    # 原始数据的csv文件
    #originDataFile = 'F:\Code\bagging-light-classification\\originData.csv'  #注意：该csv文件没有'ID'列
    originDataFile = 'originData.csv'  #注意：该csv文件没有'ID'列
    # 生成的训练集csv文件和验证集csv文件
    #trainDataFile = 'F:\Code\bagging-light-classification\\train.csv'  #注意：该csv文件没有'ID'列
    #validationDataFile = 'F:\Code\bagging-light-classification\\validation.csv'  #注意：该csv文件没有'ID'列
    trainDataFile = 'train.csv'  #注意：该csv文件没有'ID'列
    validationDataFile = 'validation.csv'  #注意：该csv文件没有'ID'列
    '''
    
    '''
    # --------------------- 如果已经生成了train.csv和validation.csv就没有必要执行下面两行程序 -----------------
    print('生成训练集的csv文件和验证集的csv文件')  
    originDict = setDict_train_validation(originDataFile)
    setCSV_train_validation(originDict, trainDataFile, validationDataFile)
    '''
    
    # --------------------- step2. 生成训练集和验证集的数据字典 ---------------------
    '''
    print('生成训练数据字典')
    # 生成训练集的数据字典：里面共有5个采样{'1SVCFeature, 1SVCClass', ```, '5SVCFeature, 5SVCClass'}
    trainDict = loadTrainDataDict(trainDataFile)
    '''

    validationDataFile = 'validation.csv'  #注意：该csv文件没有'ID'列
    print('生成验证数据字典')
    # 生成验证集的数据字典：{ validationFeature, validationClass }
    validationDict = loadValidationDataDict(validationDataFile)
    
    '''
    # --------------------- step3. 训练模型，bagging（5个SVC） ---------------------
    print('start training Bagging model')
    bagging_SVC_model(trainDict, validationDict)
    '''
    
    # --------------------- step4. 加载模型，bagging（5个SVC），对验证集进行验证 ---------------------
    print('验证模型')
    predictClass_new(validationDict)
    
# 在python编译器读取源文件的时候会执行它找到的所有代码，
# 而在执行之前会根据当前运行的模块是否为主程序而定义变量__name__的值为__main__还是模块名。
# 因此，该判断语句为真的时候，说明当前运行的脚本为主程序，而非主程序所引用的一个模块。
# 这在当你想要运行一些只有在将模块当做程序运行时而非当做模块引用时才执行的命令，只要将它们放到if __name__ == "__main__:"判断语句之后就可以了。
if __name__ == '__main__':  
    start = time()
    # print('开始训练模型\n')
    
    main()
    
    print('\n')
    # print('训练结束')
    end = time()
    print('用时 %.5f seconds.' % (end-start))








# In[ ]:




