# 机器学习——02 k-近邻算法

## 参考资料

1. [AIlearning](https://ailearning.apachecn.org/#/docs/ml/2)
2. [Machine-Learning-in-Action](https://github.com/TeFuirnever/Machine-Learning-in-Action)
3. 庞善民.西安交通大学机器学习导论2022春PPT

## 一、KNN概述

**k-近邻（kNN, k-NearestNeighbor）算法**是一种基本分类与回归方法，这里只讨论分类问题中的 k-近邻算法。

k 近邻算法的输入为实例的特征向量，对应于特征空间的点；输出为实例的类别，可以取多类。k 近邻算法假设给定一个训练数据集，其中的实例类别已定。分类时，对新的实例，根据其 k 个最近邻的训练实例的类别，通过多数表决等方式进行预测。因此，k近邻算法不具有显式的学习过程。

k 近邻算法实际上利用训练数据集对特征向量空间进行划分，并作为其分类的“模型”。 **k值的选择、距离度量以及分类决策规则**是k近邻算法的三个基本要素。

KNN算法本身简单有效，它是一种lazy- learning算法

## 二、应用场景

电影可以按照题材分类，那么如何区分 `动作片` 和 `爱情片` 呢？

1. 动作片: 打斗次数更多
2. 爱情片: 亲吻次数更多

基于电影中的亲吻、打斗出现的次数，使用 k-近邻算法构造程序，就可以自动划分电影的题材类型。

| 电影名称                   | 打斗镜头 | 接吻镜头 | 电影类型 |
| -------------------------- | -------- | -------- | -------- |
| California Man             | 3        | 104      | 爱情片   |
| He's Not Really into Dudes | 2        | 100      | 爱情片   |
| Beautiful Woman            | 1        | 81       | 爱情片   |
| Kevin Longblade            | 101      | 10       | 动作片   |
| Robo Slayer 3000           | 99       | 5        | 动作片   |
| Amped Ⅱ                    | 98       | 2        | 动作片   |
| ？                         | 18       | 90       | 未知     |

<p align="center">table 1 : 每部电影的打斗镜头数、接吻镜头数以及电影评估类型
电影名称

| 电影名称                   | 与未知电影的距离 |
| -------------------------- | ---------------- |
| California Man             | 20.5             |
| He's Not Really into Dudes | 18.7             |
| Beautiful Woman            | 19.2             |
| Kevin Longblade            | 115.3            |
| Robo Slayer 3000           | 117.4            |
| Amped Ⅱ                    | 118.9            |

<p align="center">table 2 : 已知电影与未知电影的距离

现在根据上面我们得到的样本集中所有电影与未知电影的距离，按照距离递增排序，可以找到 k 个距离最近的电影。 假定 k=3，则三个最靠近的电影依次是， He's Not Really into Dudes 、 Beautiful Woman 和 California Man。 knn 算法按照距离最近的三部电影的类型，决定未知电影的类型，而这三部电影全是爱情片，因此我们判定未知电影是爱情片。

## 三、KNN原理

### 分类原理

对一个未知样本进行分类：

> :one: 计算未知样本与标记样本的距离(最废时)
>
> :two: 确定k个近邻（超参，不鲁棒）
>
> :three: 使用近邻样本的标签确定目标的标签：例如，**将其划分到k个样本中出现最频繁的类**

### 通俗解释

给定一个训练数据集，对新的输入实例，在训练数据集中找到与该实例最邻近的 k 个实例，这 k 个实例的多数属于某个类，就把该输入实例分为这个类。

### 开发流程

1. 收集数据: 任何方法 
2. 准备数据: 距离计算所需要的数值，最好是结构化的数据格式 
3. 分析数据: 任何方法 
4. 训练算法: 此步骤不适用于 k-近邻算法 
5. 测试算法: 计算错误率 
6. 使用算法: 输入样本数据和结构化的输出结果，然后运行 k-近邻算法判断输入数据分类属于哪个分类，最后对计算出的分类执行后续处理

### 算法特点

优点: 精度高、对异常值不敏感、无数据输入假定 

缺点: 计算复杂度高、空间复杂度高 

适用数据范围: 数值型和标称型

## 四、实际项目案例——优化约会网站的配对效果

### 项目概述

海伦使用约会网站寻找约会对象。经过一段时间之后，她发现曾交往过三种类型的人:

- 不喜欢的人
- 魅力一般的人
- 极具魅力的人

她希望:

1. 工作日与魅力一般的人约会
2. 周末与极具魅力的人约会
3. 不喜欢的人则直接排除掉

现在她收集到了一些约会网站未曾记录的数据信息，这更有助于匹配对象的归类。

### 开发流程

收集数据: 提供文本文件 

准备数据: 使用 Python 解析文本文件 

分析数据: 使用 Matplotlib 画二维散点图 

训练算法: 此步骤不适用于 k-近邻算法 

测试算法: 使用海伦提供的部分数据作为测试样本。        

测试样本和非测试样本的区别在于: 测试样本是已经完成分类的数据，如果预测分类与实际类别不同，则标记为一个错误。 

使用算法: 产生简单的命令行程序，然后海伦可以输入一些特征数据以判断对方是否为自己喜欢的类型。

#### 收集数据

海伦把这些约会对象的数据存放在文本文件 [datingTestSet2.txt](https://ailearning.apachecn.org/#/data/2.KNN/datingTestSet2.txt) 中，总共有 1000 行。海伦约会的对象主要包含以下 3 种特征:

- 每年获得的飞行常客里程数
- 玩视频游戏所耗时间百分比
- 每周消费的冰淇淋公升数

文本文件数据格式如下:

```
40920    8.326976    0.953952    3
14488    7.153469    1.673904    2
26052    1.441871    0.805124    1
75136    13.147394   0.428964    1
38344    1.669788    0.134296    1
```

#### 准备数据

将文本记录转换为 NumPy 的解析程序

```python
def fileToMatrix(filename):
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
            dataMat[index, : ] = listData[:3]
            dataLabel.append(int(listData[-1]))
            index += 1
    return dataMat, dataLabel
```

#### 分析数据

使用 Matplotlib 画二维散点图

scatter-散点图常用参数

![img](https://pic1.zhimg.com/80/v2-bc0e7259d6517b921e3e08b16a33611c_720w.webp)

```python
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
    #画出散点图,以dataMat矩阵的第一(飞行常客例程)、第二列(玩游戏)数据画散点数据,散点大小为15
    ax.scatter(x=dataMat[:,0], y=dataMat[:,1], color=LabelsColors,s=15)
    #设置标题,x轴label,y轴label
    ax_title_text = ax.set_title(u'Frequent flier miles earned per year versus time spent playing video games')
    ax_xlabel_text = ax.set_xlabel(u'Frequent flyer miles earned per year')
    ax_ylabel_text = ax.set_ylabel(u'Percentage of time spent playing video games')
    plt.setp(ax_title_text, size=9, color='red')
    plt.setp(ax_xlabel_text, size=8, color='black')
    plt.setp(ax_ylabel_text, size=8, color='black')

    
    #设置图例
    didntLike = mlines.Line2D([], [], color='black', marker='.',
                      markersize=6, label='Did Not Like')
    smallDoses = mlines.Line2D([], [], color='orange', marker='.',
                      markersize=6, label='Like in Small Doses')
    largeDoses = mlines.Line2D([], [], color='red', marker='.',
                      markersize=6, label='Like in Large Doses')
    #添加图例
    ax.legend(handles=[didntLike,smallDoses,largeDoses])
    #显示图片
    plt.show()
```

下图中采用矩阵的第一和第二列属性得到很好的展示效果，清晰地标识了三个不同的样本分类区域，具有不同爱好的人其类别区域也不同。

![Figure_1](https://note-image-1307786938.cos.ap-beijing.myqcloud.com/typora/Figure_1.png)

**归一化数据**：归一化是一个让权重变为统一的过程

| 序号 | 玩视频游戏所耗时间百分比 | 每年获得的飞行常客里程数 | 每周消费的冰淇淋公升数 | 样本分类 |
| ---- | ------------------------ | ------------------------ | ---------------------- | -------- |
| 1    | 0.8                      | 400                      | 0.5                    | 1        |
| 2    | 12                       | 134 000                  | 0.9                    | 3        |
| 3    | 0                        | 20 000                   | 1.1                    | 2        |
| 4    | 67                       | 32 000                   | 0.1                    | 2        |

样本3和样本4的距离:
$$
\sqrt{(0-67)^2 + (20000-32000)^2 + (1.1-0.1)^2 }
$$


归一化就是要把需要处理的数据经过处理后（通过某种算法）限制在需要的一定范围内。首先归一化是为了后面数据处理的方便，其次是保正程序运行时收敛加快。方法有如下:

![img](https://picx.zhimg.com/80/v2-334b1b075e35eeb3f1263f2a481f019f_1440w.webp?source=1940ef5c)

在统计学中，归一化的具体作用是归纳统一样本的统计分布性。归一化在0-1之间是统计的概率分布，归一化在-1--+1之间是统计的坐标分布。

```python
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
```

#### 训练算法

因为测试数据每一次都要与全量的训练数据进行比较，所以这个过程是没有必要的。

算法步骤：

对于每一个在数据集中的数据点:     

1. 计算目标的数据点（需要分类的数据点）与该数据点的距离    
2. 将距离排序: 从小到大    
3. 选取前K个最短距离    
4. 选取这K个中最多的分类类别    
5. 返回该类别来作为目标数据点的预测值

欧氏距离其实就是L2范数，数学定义如下：
$$
d_{12}=\sqrt{\sum_{k=1}^n(x_{1k}-x_{2k})^2}
$$

```python
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
```

#### 测试算法

测试算法: 使用海伦提供的部分数据作为测试样本。如果预测分类与实际类别不同，则标记为一个错误。

```python
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
        classifierResult = knn_classify(
            normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 3)
        if (classifierResult != datingLabels[i]):
            errorCount += 1.0
    print("the total error rate is: %f" % (errorCount / float(numTestVecs)))

```

```
the total error rate is: 0.050000
```



#### 使用算法

 产生简单的命令行程序，然后海伦可以输入一些特征数据以判断对方是否为自己喜欢的类型。

```python
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
```



运行结果

```
每年获得的飞行常客里程数:50000
玩视频游戏所耗时间百分比:8
每周消费的冰激淋公升数:1
你可能非常喜欢这个人
```

## 五、KNN小结

KNN 是一个简单的无显示学习过程，非泛化学习的监督学习模型。在分类和回归中均有应用。

### 基本原理

简单来说: 通过距离度量来计算查询点（query point）与每个训练数据点的距离，然后选出与查询点（query point）相近的K个最邻点（K nearest neighbors），使用分类决策来选出对应的标签来作为该查询点的标签。

### KNN三要素

#### k的取值

对查询点标签影响显著（效果拔群）。k值小的时候近似误差小，估计误差大。 k值大近似误差大，估计误差小。

> 近似误差其实可以理解为模型估计值与实际值之间的差距。
>
> 估计误差其实可以理解为模型的估计系数与实际系数之间的差距。

如果选择较小的 k 值，就相当于用较小的邻域中的训练实例进行预测，“学习”的近似误差（approximation error）会减小，只有与输入实例较近的（相似的）训练实例才会对预测结果起作用。但缺点是“学习”的估计误差（estimation error）会增大，预测结果会对近邻的实例点非常敏感。如果邻近的实例点恰巧是噪声，预测就会出错。换句话说，**k 值的减小就意味着整体模型变得复杂，容易发生过拟合**。

如果选择较大的 k 值，就相当于用较大的邻域中的训练实例进行预测。其优点是可以减少学习的估计误差。但缺点是学习的近似误差会增大。这时与输入实例较远的（不相似的）训练实例也会对预测起作用，使预测发生错误。 **k 值的增大就意味着整体的模型变得简单**。

太大太小都不太好，可以用**交叉验证（cross validation）**来选取适合的k值。

#### 距离度量 Metric/Distance Measure

距离度量通常为欧式距离（Euclidean distance），还可以是 Minkowski 距离或者曼哈顿距离。也可以是地理空间中的一些距离公式。

#### 分类决策 （decision rule）

分类决策在分类问题中 通常为通过少数服从多数 来选取票数最多的标签，在回归问题中通常为 K个最邻点的标签的平均值

