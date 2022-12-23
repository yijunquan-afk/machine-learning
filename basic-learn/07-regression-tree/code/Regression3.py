from numpy import *
from RegressionTree1 import *
from RegressionTree2 import *


def regTreeEval(model, inputData):
    """对回归树进行预测

    Args:
        model : 指定模型，可选值为回归树模型或模型树模型，这里为回归树
        inputData: 输入的测试数据

    Returns:
        float: 将输入的模型数据转换为浮点数返回
    """    
    return float(model)


def modelTreeEval(model, inputData):
    """对模型树进行预测

    Args:
        model : 指定模型，可选值为回归树模型或模型树模型，这里为模型树
        inputData: 输入的测试数据

    Returns:
        float: 将测试数据乘以回归系数得到一个预测值 ，转化为浮点数返回
    """    
    n = shape(inputData)[1]
    X = mat(ones((1, n+1)))
    X[:, 1:n+1] = inputData
    return float(X*model)


def treeForeCast(tree, inputDataa, modelEval=regTreeEval):
    """对特定模型的树进行预测，可以是回归树也可以是模型树

    Args:
        tree: 已经训练好的树的模型
        inputData: 输入的测试数据
        modelEval :  预测的树的模型类型. Defaults to regressionTreeEval.

    Returns:
        float: 预测值
    """        
    if not isTree(tree):
        return modelEval(tree, inputDataa)
    if inputDataa[tree['splitIndex']] > tree['splitValue']:
        if isTree(tree['left']):
            return treeForeCast(tree['left'], inputDataa, modelEval)
        else:
            return modelEval(tree['left'], inputDataa)
    else:
        if isTree(tree['right']):
            return treeForeCast(tree['right'], inputDataa, modelEval)
        else:
            return modelEval(tree['right'], inputDataa)


def createForeCast(tree, testData, modelEval=regTreeEval):
    """调用 treeForeCast ，对特定模型的树进行预测，可以是回归树也可以是模型树
    Args:
        tree: 已经训练好的树的模型
        inData: 输入的测试数据
        modelEval:预测的树的模型类型，可选值为 regTreeEval（回归树） 或 modelTreeEval（模型树），默认为回归树
    Returns:
        返回预测值矩阵
    """    
    m = len(testData)
    yHat = mat(zeros((m, 1)))
    for i in range(m):
        yHat[i, 0] = treeForeCast(tree, mat(testData[i]), modelEval)
    return yHat


if __name__ == '__main__':
    trainMat = mat(loadData("data/bikeSpeedVsIq_train.txt"))
    testMat = mat(loadData("data/bikeSpeedVsIq_test.txt"))
    myTree = createTree(trainMat, ops=(1, 20))
    yHat = createForeCast(myTree, testMat[:, 0])
    # 返回 Pearson product-moment 相关系数。
    print(corrcoef(yHat, testMat[:, 1], rowvar=0)[0, 1])
