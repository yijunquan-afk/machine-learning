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


def plotData(data):
    """  
    绘制数据集
    """
    xcord = []
    ycord = []  # 样本点
    for i in range(len(data)):
        xcord.append(data[i][0])
        ycord.append(data[i][1])  # 样本点
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord, ycord, s=20, c='red', label='data')
    plt.title('Visualized Data')
    plt.ylabel('x2')
    plt.show()





def binSplitData(dataMat, feature, value):
    """二分数据

    Args:
        dataMat (矩阵): 矩阵化的数据
        feature (特征索引): 待切分的特征
        value (数值): 特征的某个值

    Returns:
        mat0, mat1: 切分后的数据集矩阵
    """
    # nonzero()[0]: 返回满足条件的行索引
    mat0 = dataMat[nonzero(dataMat[:, feature] > value)[0], :]
    mat1 = dataMat[nonzero(dataMat[:, feature] <= value)[0], :]
    return mat0, mat1

def regLeaf(data):
    """  
    返回每一个叶子结点的均值
    regLeaf 是产生叶节点的函数，就是求均值，即用聚类中心点来代表这类数据
    """
    return mean(data[:, -1])


def regErr(data):
    """  
    计算总方差=方差*样本数
    求这组数据的方差，即通过决策树划分，可以让靠近的数据分到同一类中去
    """
    return var(data[:, -1]) * shape(data)[0]


def chooseBestSplit(dataMat, leafType=regLeaf, errType=regErr, ops=(1, 4)):
    """用最佳方式切分数据集 和 生成相应的叶节点

    Args:
        dataMat (矩阵): 矩阵化的数据集
        leafType (函数, optional): 建立叶子点的函数. Defaults to regLeaf.
        errType (函数, optional):误差计算函数(求总方差). Defaults to regErr.
        ops (tuple, optional): [容许误差下降值，切分的最少样本数]。. Defaults to (1, 4).

    Returns:
        bestIndex: feature的index坐标
        bestValue: 切分的最优值
    """
    tolS = ops[0]  # 最小误差下降值，划分后的误差减小小于这个差值，就不用继续划分
    tolN = ops[1]  # 划分最小size

    if len(set(dataMat[:, -1].T.tolist()[0])) == 1:
        # 如果集合size为1，也就是说全部的数据都是同一个类别，不用继续划分。
        return None, leafType(dataMat)

    m, n = shape(dataMat)
    # 无分类误差的总方差和
    S = errType(dataMat)
    bestS, bestIndex, bestValue = inf, 0, 0
    # 循环处理每一列对应的feature值
    for featIndex in range(n - 1):
        for splitVal in set(dataMat[:, featIndex].T.tolist()[0]):
            # 对该列进行分组，然后组内的成员的val值进行 二元切分
            mat0, mat1 = binSplitData(dataMat, featIndex, splitVal)
            # 判断二元切分的方式的元素数量是否符合预期
            # 小于划分最小size
            if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):
                continue
            newS = errType(mat0) + errType(mat1)
            # 如果二元切分，算出来的误差在可接受范围内，那么就记录切分点，并记录最小误差
            # 如果划分后误差小于 bestS，则说明找到了新的bestS
            if newS < bestS:
                bestIndex = featIndex
                bestValue = splitVal
                bestS = newS
    # 判断二元切分的方式的元素误差是否符合预期
    # 如果减少量(S-bestS)小于阈值，则不做分割。
    if (S - bestS) < tolS:
        return None, leafType(dataMat) 
    #根据最佳的切分特征和特征值切分数据集合               
    mat0, mat1 = binSplitData(dataMat, bestIndex, bestValue)
    # 对整体的成员进行判断，是否符合预期
    # 如果集合的 size 小于 tolN 
    if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN): 
        # 当最佳划分后，集合过小，也不划分，产生叶节点
        return None, leafType(dataMat)
    return bestIndex, bestValue

def isTree(obj):
    """  
    判断节点是否为一棵树（字典）
    """
    return(type(obj).__name__ == 'dict')

def getMean(tree):
    """从上往下遍历树直到叶节点为止，如果找到两个叶节点则计算它们的平均值。
        对 tree 进行塌陷处理，即返回树平均值。
    Args:
        tree: 输入的树
    Returns:
        返回 tree 节点的平均值
    """

    if isTree(tree['right']):
        tree['right'] = getMean(tree['right'])
    if isTree(tree['left']):
        tree['left'] = getMean(tree['left'])
    return (tree['left']+tree['right'])/2.0

def prune(tree, testData):
    """从上而下找到叶节点，用测试数据集来判断将这些叶节点合并是否能降低测试误差
    Args:
        tree : 待剪枝的树
        testData: 剪枝所需要的测试数据 testData 
    Returns:
        tree: 剪枝完成的树
    """
    # 判断是否测试数据集没有数据，如果没有，就直接返回tree本身的均值
    if shape(testData)[0] == 0:
        return getMean(tree)

    # 判断分枝是否是dict字典，如果是就将测试数据集进行切分
    if (isTree(tree['right']) or isTree(tree['left'])):
        lSet, rSet = binSplitData(testData, tree['splitIndex'], tree['splitValue'])
    # 如果是左边分枝是字典，就传入左边的数据集和左边的分枝，进行递归
    if isTree(tree['left']):
        tree['left'] = prune(tree['left'], lSet)
    # 如果是右边分枝是字典，就传入左边的数据集和左边的分枝，进行递归
    if isTree(tree['right']):
        tree['right'] = prune(tree['right'], rSet)

    # 上面的一系列操作本质上就是将测试数据集按照训练完成的树拆分好，对应的值放到对应的节点

    # 如果左右两边同时都不是dict字典，也就是左右两边都是叶节点，而不是子树了，那么分割测试数据集。
    # 1. 如果正确 
    #   * 那么计算一下总方差 和 该结果集的本身不分枝的总方差比较
    #   * 如果 合并的总方差 < 不合并的总方差，那么就进行合并
    # 注意返回的结果:  如果可以合并，原来的dict就变为了 数值
    if not isTree(tree['left']) and not isTree(tree['right']):
        lSet, rSet = binSplitData(testData, tree['splitIndex'], tree['splitValue'])
        errorNoMerge = sum(power(lSet[:, -1] - tree['left'], 2)) + sum(power(rSet[:, -1] - tree['right'], 2))
        treeMean = (tree['left'] + tree['right'])/2.0
        errorMerge = sum(power(testData[:, -1] - treeMean, 2))
        # 如果 合并的总方差 < 不合并的总方差，那么就进行合并
        if errorMerge < errorNoMerge:
            print("merging")
            return treeMean
        else:
            return tree
    else:
        return tree    



def createTree(dataMat, leafType=regLeaf, errType=regErr, ops=(1, 4)):
    """createTree(获取回归树)
    Args:
        dataMat      加载的矩阵数据
        leafType     建立叶子点的函数
        errType      误差计算函数
        ops=(1, 4)   [容许误差下降值，切分的最少样本数]
    Returns:
        retTree    决策树最后的结果
    """
    # 选择最好的切分方式:  feature索引值，最优切分值
    # choose the best split
    feat, val = chooseBestSplit(dataMat, leafType, errType, ops)
    # 如果 splitting 达到一个停止条件，那么返回 val
    if feat is None:
        return val
    retTree = {}
    retTree['splitIndex'] = feat
    retTree['splitValue'] = val
    # 大于在右边，小于在左边，分为2个数据集
    lSet, rSet = binSplitData(dataMat, feat, val)
    # 递归的进行调用，在左右子树中继续递归生成树
    retTree['left'] = createTree(lSet, leafType, errType, ops)
    retTree['right'] = createTree(rSet, leafType, errType, ops)
    return retTree

if __name__ == '__main__':
    plotData(loadData('data/data1.txt'))
