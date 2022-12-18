from numpy import *
import matplotlib.pyplot as plt
import Regression1


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


def lwlr(testP, X, y, k=1.0):
    """局部加权线性回归，在待预测点附近的每个点赋予一定的权重，
    在子集上基于最小均方差来进行普通的回归。

    Args:
        testP (行向量): 测试样本点
        X: 样本的特征数据
        y: 每个样本对应的类别标签，即目标变量
        k: 控制核函数的衰减速率. Defaults to 1.0.
    Returns:
        testP * ws: 数据点与具有权重的系数相乘得到的预测点
    Notes:
        这其中会用到计算权重的公式，w = e^((x^((i))-x) / -2k^2)
        理解: x为某个预测点，x^((i))为样本点，样本点距离预测点越近，贡献的误差越大（权值越大），越远则贡献的误差越小（权值越小）。
        关于预测点的选取，在我的代码中取的是样本点。其中k是带宽参数，控制w（钟形函数）的宽窄程度，类似于高斯函数的标准差。
        算法思路: 假设预测点取样本点中的第i个样本点（共m个样本点），遍历1到m个样本点（含第i个），算出每一个样本点与预测点的距离，
        也就可以计算出每个样本贡献误差的权值，可以看出w是一个有m个元素的向量（写成对角阵形式）。
    """
    # 转化为矩阵
    X_mat = mat(X)
    y_mat = mat(y).T
    # 数据行数
    m = shape(X_mat)[0]
    # eye()返回一个对角线元素为1，其他元素为0的二维数组，创建权重矩阵weights，
    # 该矩阵为每个样本点初始化了一个权重
    weights = mat(eye((m)))
    for i in range(m):
        # 计算 testPoint 与输入样本点之间的距离，然后下面计算出每个样本贡献误差的权值
        diffMat = testP - X_mat[i, :]
        # k控制衰减的速度
        weights[i, i] = exp(diffMat * diffMat.T / (-2.0 * k**2))
    xTx = X_mat.T * (weights * X_mat)
    if linalg.det(xTx) == 0.0:
        print("无法求得矩阵的逆")
        return
    ws = xTx.I * (X_mat.T * (weights * y_mat))  # 计算回归系数
    return testP * ws


def lwlrTest(testArr, xArr, yArr, k=1.0):
    '''测试局部加权线性回归，对数据集中每个点调用 lwlr() 函数
        Args: 
            testArr: 测试所用的所有样本点
            xArr: 样本的特征数据，即 feature
            yArr: 每个样本对应的类别标签，即目标变量
            k: 控制核函数的衰减速率
        Returns: 
            yHat: 预测点的估计值
    '''
    # 得到样本点的总数
    m = shape(testArr)[0]
    # 构建一个全部都是 0 的 1 * m 的矩阵
    yHat = zeros(m)
    # 循环所有的数据点，并将lwlr运用于所有的数据点
    for i in range(m):
        yHat[i] = lwlr(testArr[i], xArr, yArr, k)
    # 返回估计值
    return yHat


def rssError(yArr, yHatArr):
    '''计算分析预测误差的大小
        Args:
            yArr: 真实的目标变量
            yHatArr: 预测得到的估计值
        Returns:
            计算真实值和估计值得到的值的平方和作为最后的返回值
    '''
    return ((yArr - yHatArr)**2).sum()

def abaloneTest():
    '''预测鲍鱼的年龄
    Args:
        None
    Returns:
        None
    '''
    abX, abY = loadData('data/abalone.txt')
    print('训练集与测试集相同:局部加权线性回归,核k的大小对预测的影响:')
    yHat01 = lwlrTest(abX[0:99], abX[0:99], abY[0:99], 0.1)
    yHat1 = lwlrTest(abX[0:99], abX[0:99], abY[0:99], 1)
    yHat10 = lwlrTest(abX[0:99], abX[0:99], abY[0:99], 10)
    print('k=0.1时,误差大小为:',rssError(abY[0:99], yHat01.T))
    print('k=1  时,误差大小为:',rssError(abY[0:99], yHat1.T))
    print('k=10 时,误差大小为:',rssError(abY[0:99], yHat10.T))
    print('')
    print('训练集与测试集不同:局部加权线性回归,核k的大小是越小越好吗？更换数据集,测试结果如下:')
    yHat01 = lwlrTest(abX[100:199], abX[0:99], abY[0:99], 0.1)
    yHat1 = lwlrTest(abX[100:199], abX[0:99], abY[0:99], 1)
    yHat10 = lwlrTest(abX[100:199], abX[0:99], abY[0:99], 10)
    print('k=0.1时,误差大小为:',rssError(abY[100:199], yHat01.T))
    print('k=1  时,误差大小为:',rssError(abY[100:199], yHat1.T))
    print('k=10 时,误差大小为:',rssError(abY[100:199], yHat10.T))
    print('')
    print('训练集与测试集不同:简单的线性归回与k=1时的局部加权线性回归对比:')
    print('k=1时,误差大小为:', rssError(abY[100:199], yHat1.T))
    ws = Regression1.regression(abX[0:99], abY[0:99])
    yHat = mat(abX[100:199]) * ws
    print('简单的线性回归误差大小:', rssError(abY[100:199], yHat.T.A))

abaloneTest()