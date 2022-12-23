from numpy import *
import matplotlib.pyplot as plt


def loadData(fileName):
    """解析每一行，并转化为float类型

    Args:
        fileName: 文件名

    Returns:
        data: 每一行的数据集为array类型
    """
    # 假定最后一列是结果值
    data = []
    with open(fileName) as f:
        for line in f.readlines():
            currentLine = line.strip().split('\t')
            # map将currentLine中的每一个元素应用于float，返回一个列表
            floatLine = list(map(float, currentLine))
            data.append(floatLine)
    return data



def linearSolve(dataMat):
    """将数据集格式化成目标变量Y和自变量X，执行简单的线性回归，得到ws
    Args:
        dataMat: 输入数据矩阵
    Returns:
        ws: 执行线性回归的回归系数 
        X : 格式化自变量X
        Y : 格式化目标变量Y
    """
    m, n = shape(dataMat)
    # 产生一个关于1的矩阵
    X = mat(ones((m, n)))
    Y = mat(ones((m, 1)))
    # X的0列为1，常数项，用于计算平衡误差
    X[:, 1: n] = dataMat[:, 0: n-1]
    Y = dataMat[:, -1]

    # 转置矩阵*矩阵
    xTx = X.T * X
    # 如果矩阵的逆不存在，会造成程序异常
    if linalg.det(xTx) == 0.0:
        raise NameError('矩阵不可逆')
    # 最小二乘法求最优解:  w0*1+w1*x1=y
    ws = xTx.I * (X.T * Y)
    return ws, X, Y

def modelLeaf(dataMat):
    """数据不再需要切分的时候，生成叶节点的模型。
    f(x) = x0 + x1 * featrue1+ x3 * featrue2 ...
    Args:
        dataMat: 输入数据集
    Returns:
        调用 linearSolve 函数，返回得到的 回归系数ws
    """
    ws, X, Y = linearSolve(dataMat)
    return ws

def modelErr(dataMat):
    """在给定数据集上计算误差。
    Args:
        dataMat: 输入数据矩阵
    Returns:
        调用 linearSolve 函数，返回 yHat 和 Y 之间的平方误差。
    """
    ws, X, Y = linearSolve(dataMat)
    yHat = X * ws
    return sum(power(Y - yHat, 2))