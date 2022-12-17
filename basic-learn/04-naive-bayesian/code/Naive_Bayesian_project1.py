from numpy import *


def loadDataSet():
    """创建数据集

    Returns:
        postingList: 单词列表
        classVec   : 所属类别
    """
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],  # [0,0,1,1,1......]
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak',
                       'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]  # 1为正例
    return postingList, classVec


def createVocabularyList(dataSet):
    """获取所有单词的集合

    Args:
        dataSet: 数据集
    Returns:
        不含重复元素的单词列表
    """
    voSet = set([])
    for document in dataSet:
        voSet = voSet | set(document)
    return list(voSet)


def bagofWords(voList, document):
    """使用词袋模型将文本转换为向量

    Args:
        voList: 词汇表
        document: 输入的句子

    Returns:
        returnVec: 单词向量
    """
    returnVec = zeros(len(voList))
    # 遍历文档中的所有单词，如果出现了词汇表中的单词，则将输出的文档向量中的对应值设为1
    for word in document:
        if word in voList:
            returnVec[voList.index(word)] = 1
        else:
            print("the word: %s is not in my Vocabulary!" % word)
    return returnVec


def trainNB(trainMatrix, trainLabel):
    """训练数据

    Args:
        trainMatrix (矩阵): 文档单词矩阵 [[1,0,1,1,1....],[],[]...]
        trainLabel (向量): 文档类别矩阵 [0,1,1,0....]
    """
    # 总文件数
    numTrainDocs = len(trainMatrix)
    # 总单词数
    numWords = len(trainMatrix[0])

    # 侮辱性文件的出现概率，即trainLabel中所有的1的个数，
    # 使用拉普拉斯平滑校正
    pc_1 = sum(trainLabel) / float(numTrainDocs)

    # 构造单词出现次数列表
    p0Num = zeros(numWords)  
    p1Num = zeros(numWords)

    # 整个数据集单词出现总数
    p0Denom = 0.0
    p1Denom = 0.0

    for doc in range(numTrainDocs):
        if trainLabel[doc] == 1:
            # 词在类中出现的次数
            p1Num += trainMatrix[doc]
            # 类中的总词数
            p1Denom += sum(trainMatrix[doc])
        else:
            p0Num += trainMatrix[doc]
            p0Denom += sum(trainMatrix[doc])
    # 使用log避免累乘数太小
    # 类别1的概率向量，使用拉普拉斯平滑校正
    p1Vect = log(p1Num + 1  / p1Denom + 2)
    # 类别0的概率向量，使用拉普拉斯平滑校正
    p0Vect = log(p0Num + 1 / p0Denom + 2)
    return p0Vect, p1Vect, pc_1  

def classifyNB(inputData, p0Vec, p1Vec, pClass1):
    """使用朴素贝叶斯进行分类

    Args:
        inputData (向量): 待分类的数据
        p0Vec (向量): 类别1的概率向量
        p1Vec (向量): 类别2的概率向量
        pClass1 (数): 类别1文档的出现概率

    Returns:
        数: 分类结果
    """  
    # log的使用使乘法变为家法  
    # 使用 NumPy 数组来计算两个向量相乘的结果，这里的相乘是指对应元素相乘，
    # 即先将两个向量中的第一个元素相乘，然后将第2个元素相乘，以此类推。
    # 这里的 inputData * p1Vec 的意思就是将每个词与其对应的概率相关联起来
    p1 = sum(inputData * p1Vec) + log(pClass1) # P(w|c1) * P(c1) ，即贝叶斯准则的分子
    p0 = sum(inputData * p0Vec) + log(1.0 - pClass1) # P(w|c0) * P(c0) ，即贝叶斯准则的分子·
    if p1 > p0:
        return 1
    else:
        return 0


def testingNB():
    """
    测试朴素贝叶斯算法
    """
    # 1. 加载数据集
    listOPosts, listClasses = loadDataSet()
    # 2. 创建单词集合
    myVocabList = createVocabularyList(listOPosts)
    # 3. 计算单词是否出现并创建数据矩阵
    trainMat = []
    for postinDoc in listOPosts:
        # 返回m*len(myVocabList)的矩阵， 记录的都是0，1信息
        trainMat.append(bagofWords(myVocabList, postinDoc))
    # 4. 训练数据
    p0V, p1V, pAb = trainNB(array(trainMat), array(listClasses))
    # 5. 测试数据
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = array(bagofWords(myVocabList, testEntry))
    print( testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb))
    testEntry = ['stupid', 'garbage']
    thisDoc = array(bagofWords(myVocabList, testEntry))
    print (testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb))


testingNB()