
#################################################
# Author : CIEL 
# Date   : 2017-10-20
# Function : 用平衡的数据集训练模型
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
#from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import train_test_split  
from sklearn.model_selection import StratifiedKFold  
from sklearn import svm
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
#from sklearn.learning_curve import learning_curve
#from sklearn.cross_validation import cross_val_score
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
        #for i in range(1, 343):
        for i in range(1, 341):   # 没有像素位置，只有超像素特征
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
        #for i in range(1, 343):
        for i in range(1, 341):  # 没有像素位置，只有超像素特征
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
        #for i in range(1, 343):
        for i in range(1, 341):  # 没有像素位置，只有超像素特征
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
# 函数返回一个字典：键为集合编号（trainFeature, trainClass）
def loadTrainDataDict(trainFile):
    if os.path.exists(trainFile):
        # 生成title
        listTitle = []
        listTitle.append('class')
        #for i in range(1, 343):
        for i in range(1, 341):  # 没有像素位置，只有超像素特征
            listTitle.append('feature' + str(i))
            
        # read the csv file
        # pd.read_csv: 封装在DataFrame数据结构中
        dataMatrix = np.array(pd.read_csv(trainFile, header=None, skiprows=1, names=listTitle))  # skiprows=1跳过第一行
        
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
        
        setTrainDict = {}  # 创建字典，用于存储生成的特征和类别
        setTrainDict['trainFeature'] =  np.array(sampleM)
        setTrainDict['trainClass'] = np.array(classM)

        #print ("TrainDict:")
        #print(np.array(sampleM).shape)
        #print(np.array(classM).shape)
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
        #for i in range(1, 343):
        for i in range(1, 341):  # 没有像素位置，只有超像素特征
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

        #print ("ValidationDict:")
        #print(np.array(sampleM).shape)
        #print(np.array(classM).shape)
        #print(setValidationDict)



        return setValidationDict
    else:  #读取csv文件失败
        print('No such file or directory!')  
            

#支持向量机（Support Vector Machine）:SVC
def SVC():  
    # 将probability设为True,可以计算每个样本到各个类别的概率
    # 20170926：全部采用默认参数
    #clf = svm.SVC(probability = True) 
    
    # 20171023：采用rbf核函数，C和gamma为libsvm的寻优结果
    clf = svm.SVC(kernel='rbf', random_state=0, gamma=0.5, C=32.0)
    return clf


# Utility function to report best scores
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")


# 随机搜索，使用SVM分类器
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
def RPO_SVM(trainDict):
    # 训练数据
    trainFeatureMatrix = trainDict['trainFeature']  # 特征
    trainClass = trainDict['trainClass']  # 类别

    # 构造模型
    '''
    pipeline(流水线）：跟踪记录各步骤的操作，对各步骤进行一个封装，确保代码的复杂程度不至于超出掌控范围
    输入：一连串的数据挖掘步骤，最后一步必须是估计器（estimater，即分类器），前几步是转换器（transformer，用于数据预处理和数据转换）。输入的数据集经过转换器的处理后，
    输出的结果作为下一步的输入。最后，用估计器对数据进行分类。
    每一步都用元祖（‘名称’，步骤）来表示。
    ------------------------------------------------------------------------------------------------------------------
    StandardScaler：标准化对象，用来计算训练集上的平均值和标准差（standard deviation)，同样需要在测试集上使用相同的转换
    ------------------------------------------------------------------------------------------------------------------
    SVC：random_state:数据洗牌是的种子值
    随机种子：一种以随机数作为对象的，以真随机数（种子）为初始条件的随机数。
    '''
    pipe_SVC = Pipeline([('scl', StandardScaler()), ('clf', svm.SVC(random_state=0))])   # 创建流水线
    param_range = [10**c for c in range (-4, 4)]   # 10**c：10的c次幂
    param_dist = {'clf_C':param_range, 'clf_gamma':param_range, 'clf_kernel':['linear', 'rbf']}

    # run randomized search
    '''
    RandomizedSearchCV：在一定迭代次数下，从一个随机分布中抽样选取算法参数。
    n_iter：抽样迭代次数
    '''
    n_iter_search = 20
    random_search = RandomizedSearchCV(pipe_SVC, param_distributions=param_dist, n_iter=n_iter_search)

    report(random_search.cv_results_)

    return random_search.best_estimator_


from sklearn.model_selection import GridSearchCV
# 使用网格搜索，训练SVM分类器
def GS_SVM(trainDict):
    print("start GridSearchCV ...")
    start = time()
    print('start training model ...')

    # 训练数据
    trainFeatureMatrix = trainDict['trainFeature']  # 特征


    trainClass = trainDict['trainClass']  # 类别

    '''
    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100, 1000]},
                        {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}
                        ]
    '''
    param_range = [10 ** c for c in range(-4, 4)]
    #param_grid = [{'C': param_range, 'kernel': ['linear']},  # 线性SVM只需调优正则化参数C
    #              {'C': param_range, 'gamma': param_range, 'kernel': ['rbf']}  # rbf核需要调优C和gamma
    #              ]
    param_grid = [{'C': param_range, 'gamma': param_range, 'kernel': ['rbf']}]  # rbf核需要调优C和gamma

    grid_search = GridSearchCV(SVC(), param_grid, cv=10, n_jobs=-1, verbose=1)
    grid_search.fit(trainFeatureMatrix, trainClass)

    # 分析结果自动保存
    cv_result = pd.DataFrame.from_dict(grid_search.cv_results_)
    with open('cv_result.csv', 'w') as f:
        cv_result.to_csv(f)

    end = time()
    print('用时 %.5f seconds.' % (end - start))

    best_parameters = grid_search.best_estimator_.get_params()
    for para, val in best_parameters.items():
        print(para, val)

    model = SVC(kernel='rbf', C=best_parameters['C'], gamma=best_parameters['gamma'], probability=True)
    model.fit(trainFeatureMatrix, trainClass)
    return model

    '''
    # 构造模型
    pipe_svc = Pipeline([('scl', StandardScaler()), ('clf', SVC())])
    param_range = [10**c for c in range(-4, 4)]
    param_grid = [{'clf_C': param_range, 'clf_kernel': ['linear']},  # 线性SVM只需调优正则化参数C
                  {'clf_C': param_range, 'clf_gamma': param_range, 'clf_kernel': ['rbf']}   # rbf核需要调优C和gamma
                  ]

    gs = GridSearchCV(estimator=pipe_svc, param_grid=param_grid, scoring='accuracy', cv=10, n_jobs=-1)
    gs = gs.fit(trainFeatureMatrix, trainClass)

    # 使用最优模型进行性能评估
    clf = gs.best_estimator_
    clf.fit(trainFeatureMatrix, trainClass)

    report(gs.cv_results_)

    return gs.best_estimator_
    '''

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

# SVC模型的训练
def SVC_model(trainDict, validationDict):  # 输入参数为：训练集的字典，验证集的字典
    # 获取第i个SVC分类器  
    #clf_SVC = SVC()
    clf_SVC = GS_SVM(trainDict)
    
    #SVC_rate = 0.0  # SVC_rate用于将每个SVC模型的所有识别率累加
    SVC_predict_result = []  # SVC_predict_result记录每个SVC模型对验证数据所属类别的判断
    
    # 训练数据
    #trainFeatureMatrix = trainDict['trainFeature']  #训练数据的特征
    #trainClass = trainDict['trainClass']   #训练数据的类别
    #print(trainFeatureMatrix)
    #print(trainClass)
    
    start = time()
    print('start training model ...')
    
    #训练SVC模型
    #clf_SVC.fit(trainFeatureMatrix, trainClass)

    # 保存模型
    print('saving SVC model ...')
    joblib.dump(clf_SVC, 'clf.model')
    
    end = time()
    print('用时 %.5f seconds.' % (end-start))
        
# 加载模型，并预测数据
# dataDict为验证样本的数据字典
def predictClass_new(validationDict):  
    # 验证数据
    validationFeatureMatrix = validationDict['validationFeature'] #特征
    validationClass = validationDict['validationClass'] #类别
    
    # 加载SVC模型
    print('加载SVC模型')
    SVCmodel = joblib.load('clf.model')
    
    #计算SVC模型对验证数据的类别判断结果
    SVCpredictClass = SVCmodel.predict(validationFeatureMatrix)  # predict(X): 返回X的分类（二分类，返回0和1）
    SVC_predict_result = SVCpredictClass.tolist()
    #print('SVC_predict_clf_result')
    #print(SVC_predict_result)   
    
    # ----------------------------- 模型表现 ----------------------------- 
    # classification_report()：打印分类结果报告
    print('\n')
    print('SVC classification report')
    print(classification_report(validationClass, SVC_predict_result, target_names = ['neg', 'pos']))
    SVC_rate = getRecognitionRate(SVC_predict_result, validationClass) # 验证集的识别率
    print("SVC precision:", SVC_rate)
        

def main():
    '''
    # --------------------- step1. 生成训练集的csv文件(train.csv)和验证集的csv文件(validation.csv) ---------------------
    # 原始数据的csv文件
    #originDataFile = 'F:\Code\bagging-light-classification\\originData.csv'  #注意：该csv文件没有'ID'列
    originDataFile = 'originData.csv'  #注意：该csv文件没有'ID'列
    '''
    # 生成的训练集csv文件和验证集csv文件
    #trainDataFile = 'F:\Code\bagging-light-classification\\train.csv'  #注意：该csv文件没有'ID'列
    #validationDataFile = 'F:\Code\bagging-light-classification\\validation.csv'  #注意：该csv文件没有'ID'列
    trainDataFile = '20171023BalanceTrain.csv'  #注意：该csv文件没有'ID'列
    validationDataFile = '20171023validation.csv'  #注意：该csv文件没有'ID'列
    
    '''
    # --------------------- 如果已经生成了train.csv和validation.csv就没有必要执行下面两行程序 -----------------
    print('生成训练集的csv文件和验证集的csv文件')  
    originDict = setDict_train_validation(originDataFile)
    setCSV_train_validation(originDict, trainDataFile, validationDataFile)
    '''
    
    # --------------------- step2. 生成训练集和验证集的数据字典 ---------------------
    print('生成训练数据字典')
    # 生成训练集的数据字典
    trainDict = loadTrainDataDict(trainDataFile)

    print('生成验证数据字典')
    # 生成验证集的数据字典：{ validationFeature, validationClass }
    validationDict = loadValidationDataDict(validationDataFile)
    
    # --------------------- step3. 训练模型，SVC ---------------------
    print('start training SVC model')
    SVC_model(trainDict, validationDict)
    
    # --------------------- step4. 加载SVC模型，对验证集进行验证 ---------------------
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







