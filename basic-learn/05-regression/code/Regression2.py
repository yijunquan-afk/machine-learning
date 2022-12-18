from numpy import *
import matplotlib.pyplot as plt


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


# 绘制多条局部加权回归曲线
def plotlwlrRegression():
    xArr, yArr = loadData('data/data.txt')  # 加载数据集
    yHat_1 = lwlrTest(xArr, xArr, yArr, 1.0)  # 根据局部加权线性回归计算yHat
    yHat_2 = lwlrTest(xArr, xArr, yArr, 0.01)  # 根据局部加权线性回归计算yHat
    yHat_3 = lwlrTest(xArr, xArr, yArr, 0.003)  # 根据局部加权线性回归计算yHat
    xMat = mat(xArr)  # 创建xMat矩阵
    yMat = mat(yArr)  # 创建yMat矩阵
    srtInd = xMat[:, 1].argsort(0)  # 排序，返回索引值
    xSort = xMat[srtInd][:, 0, :]
    fig, axs = plt.subplots(nrows=3,
                            ncols=1,
                            sharex=False,
                            sharey=False,
                            figsize=(10, 8))
    plt.subplots_adjust(left=None,
                        bottom=None,
                        right=None,
                        top=None,
                        hspace=0.4)
    axs[0].plot(xSort[:, 1], yHat_1[srtInd], c='red')  # 绘制回归曲线
    axs[1].plot(xSort[:, 1], yHat_2[srtInd], c='red')  # 绘制回归曲线
    axs[2].plot(xSort[:, 1], yHat_3[srtInd], c='red')  # 绘制回归曲线
    axs[0].scatter(xMat[:, 1].flatten().A[0],
                   yMat.flatten().A[0],
                   s=20,
                   c='blue',
                   alpha=.5)  # 绘制样本点
    axs[1].scatter(xMat[:, 1].flatten().A[0],
                   yMat.flatten().A[0],
                   s=20,
                   c='blue',
                   alpha=.5)  # 绘制样本点
    axs[2].scatter(xMat[:, 1].flatten().A[0],
                   yMat.flatten().A[0],
                   s=20,
                   c='blue',
                   alpha=.5)  # 绘制样本点
    # 设置标题,x轴label,y轴label
    axs0_title_text = axs[0].set_title('lwlr, k=1.0')
    axs1_title_text = axs[1].set_title('lwlr, k=0.01')
    axs2_title_text = axs[2].set_title('lwlr ,k=0.003')
    plt.setp(axs0_title_text, size=8, weight='bold', color='red')
    plt.setp(axs1_title_text, size=8, weight='bold', color='red')
    plt.setp(axs2_title_text, size=8, weight='bold', color='red')
    plt.xlabel('X')
    plt.show()


plotlwlrRegression()