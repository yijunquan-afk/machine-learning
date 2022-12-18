import matplotlib.pyplot as plt
import numpy as np

def loadData(fileName):
    """加载数据:解析以tab键分隔的文件中的浮点数

    Args:
        fileName : 数据集文件

    Returns:
        dataMat :   feature 对应的数据集
        labelMat :  feature 对应的分类标签，即类别标签
    """
    # 获取样本特征的总数，不算最后的目标变量
    numFeat = len(open(fileName).readline().split('\t')) - 1
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        # 读取每一行
        lineArr = []
        # 删除一行中以tab分隔的数据前后的空白符号
        curLine = line.strip().split('\t')
        # i 从0到2，不包括2
        for i in range(numFeat):
            # 将数据添加到lineArr List中，每一行数据测试数据组成一个行向量
            lineArr.append(float(curLine[i]))
            # 将测试数据的输入数据部分存储到dataMat 的List中
        dataMat.append(lineArr)
        # 将每一行的最后一个数据，即类别，或者叫目标变量存储到labelMat List中
        labelMat.append(float(curLine[-1]))
    return dataMat, labelMat

def rssError(yArr, yHatArr):
    '''计算分析预测误差的大小
        Args:
            yArr: 真实的目标变量
            yHatArr: 预测得到的估计值
        Returns:
            计算真实值和估计值得到的值的平方和作为最后的返回值
    '''
    return ((yArr - yHatArr)**2).sum()


def stageWise(xArr, yArr, eps=0.01, numIt=100):
    """前向逐步线性回归
    Args:
        xArr - x输入数据
        yArr - y预测数据
        eps - 每次迭代需要调整的步长
        numIt - 迭代次数
    Returns:
        returnMat - numIt次迭代的回归系数矩阵
    """
    xMat = np.mat(xArr);
    yMat = np.mat(yArr).T  # 数据集
   #数据标准化
    yMean = np.mean(yMat, axis = 0)                  #行与行操作，求均值
    yMat = yMat - yMean                              #数据减去均值
    xMeans = np.mean(xMat, axis = 0)                 #行与行操作，求均值
    xVar = np.var(xMat, axis = 0)                    #行与行操作，求方差
    xMat = (xMat - xMeans) / xVar                    #数据减去均值除以方差实现标准化
    m, n = np.shape(xMat)
    returnMat = np.zeros((numIt, n))  # 初始化numIt次迭代的回归系数矩阵
    ws = np.zeros((n, 1))  # 初始化回归系数矩阵
    wsTest = ws.copy()
    wsMax = ws.copy()
    for i in range(numIt):  # 迭代numIt次
        # print(ws.T)                             #打印当前回归系数矩阵
        lowestError = float('inf');  # 正无穷
        for j in range(n):  # 遍历每个特征的回归系数
            for sign in [-1, 1]:
                wsTest = ws.copy()
                wsTest[j] += eps * sign  # 微调回归系数
                yTest = xMat * wsTest  # 计算预测值
                rssE = rssError(yMat.A, yTest.A)  # 计算平方误差
                if rssE < lowestError:  # 如果误差更小，则更新当前的最佳回归系数
                    lowestError = rssE
                    wsMax = wsTest
        ws = wsMax.copy()
        returnMat[i, :] = ws.T  # 记录numIt次迭代的回归系数矩阵
    return returnMat


def plotstageWiseMat():
    xArr, yArr = loadData('data/abalone.txt')
    returnMat = stageWise(xArr, yArr, 0.005, 1000)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(returnMat)
    ax_title_text = ax.set_title('Forward Stepwire Regression')
    ax_xlabel_text = ax.set_xlabel('Iterations')
    ax_ylabel_text = ax.set_ylabel('Regression Coefficient')
    plt.setp(ax_title_text, size=15, weight='bold', color='red')
    plt.setp(ax_xlabel_text, size=10, weight='bold', color='black')
    plt.setp(ax_ylabel_text, size=10, weight='bold', color='black')
    plt.show()


if __name__ == '__main__':
    plotstageWiseMat()
