from numpy import *
import matplotlib.pyplot as plt

def loadData(fileName):
    """加载数据:解析以tab键分隔的文件中的浮点数

    Args:
        fileName : 数据集文件

    Returns:
        data :   feature 对应的数据集
        label :  feature 对应的分类标签，即类别标签
    """
    # data为原始数据， label为原始数据的标签
    data = []
    label = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split()
        # 为了方便计算，我们将 X0 的值设为 1.0 ，也就是在每一行的开头添加一个 1.0 作为 X0
        data.append([1.0, float(lineArr[0]), float(lineArr[1])])
        label.append(int(lineArr[2]))
    return data, label


def sigmoid(X):
    return 1.0 / (1 + exp(-X))
    

def gradAscent(data, label, alpha = 0.001, cycles = 500):
    """梯度上升法

    Args:
        data  :  feature 对应的数据集
        label :  feature 对应的分类标签，即类别标签
        alpha :  步长. Defaults to 0.001.
        cycles:  迭代次数. Defaults to 500.

    Returns:
        weights: 回归系数
    """    
    # 转化为矩阵
    XMat = mat(data)             # 转换为 NumPy 矩阵
    yMat = mat(label).transpose() 
    # m->数据量，样本数 n->特征数
    _,n = shape(XMat)
    # 生成一个长度和特征数相同的回归系数矩阵，此处n为3 -> [[1],[1],[1]]
    weights = ones((n,1))
    for i in range(cycles):    
        # m*3 的矩阵 * 3*1 的矩阵 ＝ m*1的矩阵
        f = sigmoid(XMat*weights)    
        error = (yMat - f)# 向量相减
        # 0.001* (3*m)*(m*1) 表示在每一个列上的一个误差情况，最后得出 x0, x1, x2的系数的偏移量
        weights = weights + alpha * XMat.transpose() * error # 矩阵乘法，最后得到回归系数
    return array(weights)

def stochasticGradAscent1(data, label, alpha = 0.001):
    """随机梯度上升法
       随机梯度上升一次只用一个样本点来更新回归系数
    Args:
        data  :  feature 对应的数据集
        label :  feature 对应的分类标签，即类别标签
        alpha :  步长. Defaults to 0.001.
    Returns:
        weights: 回归系数
    """   
    m,n = shape(data)
    alpha = 0.01
    # n*1的矩阵
    # 函数ones创建一个全1的数组
    weights = ones(n)   # 初始化长度为n的数组，元素全部为 1
    for i in range(m):
        # sum(data[i]*weights)为了求 f(x)的值， f(x)=a1*x1+b2*x2+..+nn*xn,
        # 此处求出的 h 是一个具体的数值，而不是一个矩阵
        h = sigmoid(sum(data[i]*weights))
        # 计算真实类别与预测类别之间的差值，然后按照该差值调整回归系数
        error = label[i] - h
        weights = weights + alpha * error * data[i]
    return weights 


def stochasticGradAscent2(data, label, cycles = 500):
    """随机梯度上升法
       随机梯度上升一次只用一个样本点来更新回归系数
    Args:
        data  :  feature 对应的数据集
        label :  feature 对应的分类标签，即类别标签
        cycles:  迭代次数. Defaults to 500.
    Returns:
        weights: 回归系数
    """   
    m,n = shape(data)
    alpha = 0.01
    # n*1的矩阵
    # 函数ones创建一个全1的数组
    weights = ones(n)   # 初始化长度为n的数组，元素全部为 1
    for j in cycles:
        # [0, 1, 2 .. m-1]
        dataIndex = range(m)
        for i in range(m):
            # i和j的不断增大，导致alpha的值不断减少，但是不为0
            alpha = 4/(1.0+j+i)+0.0001 
            # 随机产生一个 0～len()之间的一个值
            # random.uniform(x, y) 方法将随机生成下一个实数，它在[x,y]范围内,x是这个范围内的最小值，y是这个范围内的最大值。
            randIndex = int(random.uniform(0,len(dataIndex))) 
            h = sigmoid(sum(data[dataIndex[randIndex]]*weights))
            error = label[dataIndex[randIndex]] - h
            weights = weights + alpha * error * data[i]
            # 删除该下标，避免再选到
            del(dataIndex[randIndex])
    return weights 


def plotBestFit(data, label, weights):
    """画出数据集和 Logistic 回归最佳拟合直线的函数

    Args:
        data: 样本数据的特征
        label: 样本数据的类别标签，即目标变量
        weights: 回归系数
    """                                     
    dataArr = array(data)                                         #转换成numpy的array数组
    n = shape(data)[0]                                            #数据个数
    xcord1 = []; ycord1 = []                                            #正样本
    xcord2 = []; ycord2 = []                                            #负样本
    for i in range(n):                                                  #根据数据集标签进行分类
        if int(label[i]) == 1:
            xcord1.append(dataArr[i,1]); ycord1.append(dataArr[i,2])    #1为正样本
        else:
            xcord2.append(dataArr[i,1]); ycord2.append(dataArr[i,2])    #0为负样本
    fig = plt.figure()
    ax = fig.add_subplot(111)                                           #添加subplot
    ax.scatter(xcord1, ycord1, s = 20, c = 'red', marker = 's',alpha=.5,label='class1')#绘制正样本
    ax.scatter(xcord2, ycord2, s = 20, c = 'green',alpha=.5,label='class2')            #绘制负样本
    plt.title('Logistic Classify')                                                #绘制title
    #添加图例
    ax.legend()
    """
    w0*x0+w1*x1+w2*x2=f(x)
    x0最开始就设置为1叻， x2就是我们画图的y值，而f(x)被算到w0,w1,w2身上去了
    所以:  w0+w1*x+w2*y=0 => y = (-w0-w1*x)/w2   
    """
    x = arange(-3.0, 3.0, 0.1)
    y = (-weights[0]-weights[1]*x)/weights[2]
    ax.plot(x, y)
    plt.xlabel('x'); 
    plt.ylabel('y')                                   
    plt.show()   

if __name__ == '__main__':
    data, label = loadData('data/TestSet.txt')
    weights = gradAscent(data, label)
    plotBestFit(data, label, weights)