import operator
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines


def ax(filename):
    """ 
    Desc:
       导入训练数据
    parameters:
       filename: 数据文件路径
    return: 
       数据矩阵 dataMat 和对应的类别 dataLabel
    """
    with open(filename) as f:
        # 获取文本数据行数
        lines = f.readlines()
        # 生成空矩阵
        dataMat = np.zeros((len(lines), 3))
        # 数据对应的类别
        dataLabel = []
        index = 0
        for line in lines:
            # 去除空格
            line = line.strip()
            listData = line.split('\t')
            # 将每一行的数据复制到矩阵中
            dataMat[index, :] = listData[:3]
            dataLabel.append(int(listData[-1]))
            index += 1
    return dataMat, dataLabel


def visualizeData(dataMat, dataLabel):
    """ 
    Desc:
       可视化数据
    parameters:
       dataMat:   数据矩阵
       dataLabel: 数据标签
    return: 
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    LabelsColors = []
    for i in dataLabel:
        if i == 1:
            LabelsColors.append('black')
        if i == 2:
            LabelsColors.append('orange')
        if i == 3:
            LabelsColors.append('red')
    # 画出散点图,以dataMat矩阵的第一(飞行常客例程)、第二列(玩游戏)数据画散点数据,散点大小为15
    ax.scatter(x=dataMat[:, 0], y=dataMat[:, 1], color=LabelsColors, s=15)
    # 设置标题,x轴label,y轴label
    ax_title_text = ax.set_title(
        u'Frequent flier miles earned per year versus time spent playing video games')
    ax_xlabel_text = ax.set_xlabel(u'Frequent flyer miles earned per year')
    ax_ylabel_text = ax.set_ylabel(
        u'Percentage of time spent playing video games')
    plt.setp(ax_title_text, size=9, color='red')
    plt.setp(ax_xlabel_text, size=8, color='black')
    plt.setp(ax_ylabel_text, size=8, color='black')

    # 设置图例
    didntLike = mlines.Line2D([], [], color='black', marker='.',
                              markersize=6, label='Did Not Like')
    smallDoses = mlines.Line2D([], [], color='orange', marker='.',
                               markersize=6, label='Like in Small Doses')
    largeDoses = mlines.Line2D([], [], color='red', marker='.',
                               markersize=6, label='Like in Large Doses')
    # 添加图例
    ax.legend(handles=[didntLike, smallDoses, largeDoses])
    # 显示图片
    plt.show()


def normalizeData(dataSet):
    """
    归一化特征值，消除属性之间量级不同导致的影响
    :param dataSet: 数据集
    :return: 归一化后的数据集normDataSet
    归一化公式: 
        Y = (X-Xmin)/(Xmax-Xmin)
        其中的 min 和 max 分别是数据集中的最小特征值和最大特征值。该函数可以自动将数字特征值转化为0到1的区间。
    """
    # 计算每种属性的最大值、最小值、范围
    # (3,)
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    # 极差
    ranges = maxVals - minVals
    normDataSet = np.zeros(np.shape(dataSet))
    # 行数
    m = dataSet.shape[0]

    # 生成与最小值之差组成的矩阵
    # 沿Y轴复制m倍，X轴复制1倍
    normDataSet = dataSet - np.tile(minVals, (m, 1))
    # 将最小值之差除以范围组成矩阵
    normDataSet = normDataSet / np.tile(ranges, (m, 1))
    return normDataSet, ranges, minVals


def knn_classify(input, dataSet, labels, k):
    """
    Desc:
        knn算法
    parameters:
        input: 输入的待分类数据
        dataSet: 数据集
        labels: 标签
        k: 取前k个结果
    return:
        数据分类结果
    """
    dataSetSize = dataSet.shape[0]
    # 距离度量 度量公式为欧氏距离
    diffMat = np.tile(input, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5

    #将距离排序: 从小到大
    # argsort 返回从小到大的排列在数组中的索引位置，使用函数并不会改变原来数组的值。
    sortedDistIndicies = distances.argsort()
    # 选取前K个最短距离， 选取这K个中最多的分类类别
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        # 返回字典中键 `key` 对应的值，如果没有这个键，返回0
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    # python3中用items()替换python2中的iteritems()
    # key=operator.itemgetter(1)根据字典的值进行排序
    # key=operator.itemgetter(0)根据字典的键进行排序
    # 从大到小
    sortedClassCount = sorted(classCount.items(),
                              key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def dataTest():
    """
    Desc:
        对约会网站的测试方法
    parameters:
        none
    return:
        错误数
    """
    # 设置测试数据的的一个比例（训练数据集比例=1-hoRatio）
    hoRatio = 0.1  # 测试范围,一部分测试一部分作为样本
    # 从文件中加载数据
    datingDataMat, datingLabels = ax(
        'data/datingTestSet2.txt')  # load data setfrom file
    # 归一化数据
    normMat,_,_ = normalizeData(datingDataMat)
    # m 表示数据的行数，即矩阵的第一维
    m = normMat.shape[0]
    # 设置测试的样本数量， numTestVecs:m表示训练样本的数量
    numTestVecs = int(m * hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        # 对数据测试
        result = knn_classify(
            normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 3)
        if (result != datingLabels[i]):
            errorCount += 1.0
    print("the total error rate is: %f" % (errorCount / float(numTestVecs)))


def classifyPerson():
    """
    输入人的特征，返回喜爱程度
    """    
    # 输出结果
    resultList = ['不喜欢', '有些喜欢', '非常喜欢']
    # 三维特征用户输入
    ffMiles = float(input("每年获得的飞行常客里程数:"))
    precentTats = float(input("玩视频游戏所耗时间百分比:"))
    iceCream = float(input("每周消费的冰激淋公升数:"))

    # 打开并处理数据
    datingDataMat, datingLabels = ax(
        'data/datingTestSet2.txt')  # load data setfrom file
    # 训练集归一化
    normMat, ranges, minVals = normalizeData(datingDataMat)
    # 生成NumPy数组,测试集
    inArr = np.array([ffMiles, precentTats,  iceCream])
    # 测试集归一化
    norminArr = (inArr - minVals) / ranges
    # 返回分类结果
    result = knn_classify(norminArr, normMat, datingLabels, 3)
    # 打印结果
    print("你可能%s这个人" % (resultList[result-1]))


# dataTest()
classifyPerson()
