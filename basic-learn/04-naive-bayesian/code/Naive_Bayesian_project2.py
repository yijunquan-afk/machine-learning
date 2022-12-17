from numpy import *
import re


def textParse(inputString):
    '''接收一个大字符串并将其解析为字符串列表
    Args:
        inputString -- 大字符串
    Returns:
        去掉少于 2 个字符的字符串，并将所有字符串转换为小写，返回字符串列表
    '''

    # 使用正则表达式来切分句子，其中分隔符是除单词、数字外的任意字符串
    listOfTokens = re.split(r'\W+', inputString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]


def loadDataSet():
    """创建数据集

    Returns:
        postingList: 单词列表
        classVec   : 所属类别
    """
    postingList = [
        ['my', 'dog', 'has', 'flea', 'problems', 'help',
         'please'],  # [0,0,1,1,1......]
        ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
        ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
        ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
        ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
        ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']
    ]
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
    p1Vect = log(p1Num + 1 / p1Denom + 2)
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
    p1 = sum(inputData * p1Vec) + log(pClass1)  # P(w|c1) * P(c1) ，即贝叶斯准则的分子
    # P(w|c0) * P(c0) ，即贝叶斯准则的分子·
    p0 = sum(inputData * p0Vec) + log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0


def spamTest():
    """交叉验证
    """
    docList = []
    classList = []
    fullText = []
    for i in range(1, 26):
        # 切分，解析数据，并归类为 1 类别
        wordList = textParse(
            open('data/4.NaiveBayes/email/spam/%d.txt' % i,
                 'r',
                 encoding='utf-8').read())
        docList.append(wordList)
        classList.append(1)
        # 切分，解析数据，并归类为 0 类别
        wordList = textParse(
            open('data/4.NaiveBayes/email/ham/%d.txt' % i,
                 'r',
                 encoding='utf-8').read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabularyList(docList)  #创建词汇表，不重复
    trainingSet = list(range(50))
    testSet = []  #创建存储训练集的索引值的列表和测试集的索引值的列表
    for i in range(10):  #从50个邮件中，随机挑选出40个作为训练集,10个做测试集
        randIndex = int(random.uniform(0, len(trainingSet)))  #随机选取索索引值
        testSet.append(trainingSet[randIndex])  #添加测试集的索引值
        del (trainingSet[randIndex])  #在训练集列表中删除添加到测试集的索引值
    trainMat = []
    trainClasses = []  #创建训练集矩阵和训练集类别标签系向量
    for docIndex in trainingSet:  #遍历训练集
        trainMat.append(bagofWords(vocabList,
                                   docList[docIndex]))  #将生成的词集模型添加到训练矩阵中
        trainClasses.append(classList[docIndex])  #将类别添加到训练集类别标签系向量中
    p0V, p1V, pSpam = trainNB(array(trainMat), array(trainClasses))  #训练朴素贝叶斯模型
    errorCount = 0  #错误分类计数
    for docIndex in testSet:  #遍历测试集
        wordVector = bagofWords(vocabList, docList[docIndex])  #测试集的词集模型
        if classifyNB(array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:  #如果分类错误
            errorCount += 1  #错误计数加1
            print("分类错误的测试集：", docList[docIndex])
    print('错误率：%.2f%%' % (float(errorCount) / len(testSet) * 100))


spamTest()
