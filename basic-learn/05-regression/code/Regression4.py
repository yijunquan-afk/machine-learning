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

# 岭回归
def ridgeRegres(xMat, yMat, lam = 0.2):
    """
    Parameters:
        xMat - x数据集
        yMat - y数据集
        lam - 缩减系数
    Returns:
        ws - 回归系数
    """
    xTx = xMat.T * xMat
    denom = xTx + np.eye(np.shape(xMat)[1]) * lam
    if np.linalg.det(denom) == 0.0:
        print("矩阵为奇异矩阵,不能转置")
        return
    ws = denom.I * (xMat.T * yMat)
    return ws

def ridgeTest(xArr, yArr):
    """岭回归测试
    Args:
        xMat - x数据集
        yMat - y数据集
    Returns:
        wMat - 回归系数矩阵
    """
    xMat = np.mat(xArr); yMat = np.mat(yArr).T
    #数据标准化
    yMean = np.mean(yMat, axis = 0)                  #行与行操作，求均值
    yMat = yMat - yMean                              #数据减去均值
    xMeans = np.mean(xMat, axis = 0)                 #行与行操作，求均值
    xVar = np.var(xMat, axis = 0)                    #行与行操作，求方差
    xMat = (xMat - xMeans) / xVar                    #数据减去均值除以方差实现标准化
    numTestPts = 30                                  #30个不同的lambda测试
    wMat = np.zeros((numTestPts, np.shape(xMat)[1])) #初始回归系数矩阵
    for i in range(numTestPts):                      #改变lambda计算回归系数
        ws = ridgeRegres(xMat, yMat, np.exp(i - 10)) #lambda以e的指数变化，最初是一个非常小的数，
        wMat[i, :] = ws.T                            #计算回归系数矩阵
    # 30x8
    return wMat

def plotwMat():
    abX, abY = loadData('data/abalone.txt')
    redgeWeights = ridgeTest(abX, abY)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(redgeWeights)
    ax_title_text = ax.set_title('The relationship between log(lambada) and regression coefficient')
    ax_xlabel_text = ax.set_xlabel('log(lambada)')
    ax_ylabel_text = ax.set_ylabel('regression coefficient')
    plt.setp(ax_title_text, size = 10, weight = 'bold', color = 'red')
    plt.setp(ax_xlabel_text, size = 10, weight = 'bold', color = 'black')
    plt.setp(ax_ylabel_text, size = 10, weight = 'bold', color = 'black')
    plt.show()


if __name__ == '__main__':
    plotwMat()