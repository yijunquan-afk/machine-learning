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
        lineArr =[]
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
    return dataMat,labelMat   


def regression(xArr,yArr):
    '''线性回归
    Args:
        xArr : 输入的样本数据，包含每个样本数据的 feature
        yArr : 对应于输入数据的类别标签，也就是每个样本对应的目标变量
    Returns:
        ws: 回归系数
    '''

    # mat()函数将xArr，yArr转换为矩阵 mat().T 代表的是对矩阵进行转置操作
    xMat = mat(xArr)
    yMat = mat(yArr).T
    # 矩阵乘法的条件是左矩阵的列数等于右矩阵的行数
    xTx = xMat.T*xMat
    # linalg.det() 函数是用来求得矩阵的行列式的，如果矩阵的行列式为0，则这个矩阵是不可逆的，就无法进行接下来的运算                   
    if linalg.det(xTx) == 0.0:
        print("无法求得矩阵的逆")
        return
    ws = xTx.I * (xMat.T*yMat)            
    return ws


def show():
    xArr, yArr = loadData('data/data.txt')   #加载数据集
    ws = regression(xArr, yArr)          #计算回归系数
    xMat = mat(xArr)                   #创建xMat矩阵
    yMat = mat(yArr)                   #创建yMat矩阵
    xCopy = xMat.copy()                   #深拷贝xMat矩阵
    xCopy.sort(0)                         #排序
    yHat = xCopy * ws                     #计算对应的y值
    fig = plt.figure()
    ax = fig.add_subplot(111)             #添加subplot
    ax.plot(xCopy[:, 1], yHat, c = 'red') #绘制回归曲线
    # 降维后还是个矩阵，矩阵.A（等效于矩阵.getA()）变成了数组
    ax.scatter(xMat[:,1].flatten().A[0], yMat.flatten().A[0], s = 20, c = 'blue',alpha = .5) #绘制样本点
    plt.title('DataSet')                  #绘制title
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()
