
# 线性模型及其实现

## 线性回归

这部分我们使用波士顿房价数据集来说明。这个数据集包含波士顿各个区域的多个特征，比如犯罪率、一氧化氮浓度、户主的年龄信息等，综合这些信息来估计某处房子的售价。


```python
## 导入所需的数据集和包
import numpy as np
from sklearn import datasets
from sklearn.linear_model import LinearRegression
```


```python
## 加载数据集
boston = datasets.load_boston()
```


```python
## 查看数据集相关信息
print boston.DESCR
```


```python
X = boston.data
y = boston.target
```


```python
lm = LinearRegression()
```


```python
## 选取前400个数据作为训练集，剩下的作为测试集
X_train = X[:400, :]
y_train = y[:400]
X_test = X[400:,:]
y_test = y[400:]
```


```python
print "训练集的规模为 %s，测试集的规模为 %s" % (str(X_train.shape), str(X_test.shape))
```


```python
lm.fit(X_train, y_train)
```


```python
## 输出所有的参数
print lm.intercept_, lm.coef_
```

然后我们就可以用它来做预测了。


```python
y_pred = lm.predict(X_test)
```

因为是回归问题，所以我们用 mean square error (mse) 来检验拟合程度的好坏。具体来说就是用这个模型去算一下测试集的 X ，看看结果跟测试集的 y 的 mse 有多大。


```python
print 'mse = %.4f' % np.mean(np.square(y_pred - y_test))
```

## 特征选取 Feature selection

但是 LinearRegression 这个函数并不提供可供选择特征的参数。因为有些特征跟我们最终的 y 关系不大，如果硬要把它加在线性模型里，反而会造成更大的误差。

所以这里我们另外导入一个包来做这件事：


```python
import statsmodels.api as sm
```

这个包的设定跟 sklearn 略有不同。它预设的模型就是 $y = \boldsymbol{w}^T\boldsymbol{x}$ ，没有 intercept 的部分。所以我们这里要手动加上一个常数项。


```python
X_train_1 = sm.add_constant(X_train)
```

接着用 sm.OLS 来做线性回归。OLS=Ordinary Least Square 普通最小二乘。注意这里调用函数的时候要把y放前面。


```python
lm_sm = sm.OLS(y_train, X_train_1)
results = lm_sm.fit()
```


```python
print results.params
```

可以看到这里的 params 跟上面 (intercept, coef) 的数值是一样的。

那么问题来了，要如何做简单的特征选取呢？


```python
results.summary()
```

这个 summary 结果我们首先关心的是右上角的 R-squared （决定系数）。一般来说这个值是正的。假如它的值变成了负数，说明当前模型非常不适合这个数据集，需要更换别的模型。

第二张表的 coef 列就是前面的 params 。同一张表我们还需要关心的是 P>|t| 那一列。假如这里的数值大于0.05就可以考虑把相应的特征踢掉了。在里就是x3, x7, x12，对应变量名为:


```python
print boston.feature_names[np.array([2,6,11])]
```

INDUS 和 AGE 都比较好理解，B 出现在这里以我们对美国的了解肯定是不科学的。这里很有可能跟数据集的样本分布有关。因为我们划分训练集的时候是粗暴地取了前400个，而不是随机地抽取400个样本。

请根据这个结论重新划分训练集/数据集进行分析。

## 逻辑回归

这部分我们用股票数据集 Smarket 来说明，数据放在Smarket.csv中。把它放在跟这个notebook同一个文件夹下即可。


```python
from sklearn.linear_model import LogisticRegression
import pandas as pd
```

这里我们导入了另一个包 pandas，这也是一个 Python 数据分析里常用的包。用它可以一次性读入一个 csv 文件。它的功能非常强大。后面要用到的话再慢慢说。


```python
smarket = pd.read_csv('Smarket.csv')
```


```python
## 用shape函数查看数据集规模
smarket.shape
```


```python
## 查看各个 column (feature) 的名称
smarket.columns
```

这个数据集包含2001年到2005年间1250天 S&P 500 股票指数的投资回报数据。Year 表示年份，Lag1~Lag5表示过去1到5个交易日的投资回报率。Today是今日的投资回报率。Direction是市场走势，它有两类，Up和Down。另外还有一个特征是Volumn，表示前一日的成交量，单位是billion。


```python
## 查看数据的整体情况
smarket.describe()
```


```python
## 查看前面几行
smarket.head()
```


```python
## 数据类型换回 ndarray
## 这里我们不考虑 Year 这个特征，因为它的数量级跟其他特征差太多了
X = np.array(smarket.iloc[:,1:7])
y = np.array(smarket['Direction'])
```


```python
X.shape
```

同样的， sklearn 提供了直接调用 Logistic Regression 的函数 LogisticRegression()，使用方法同其他的机器学习函数一样。


```python
logit = LogisticRegression()
logit.fit(X, y)
```


```python
y_pred = logit.predict(X)
```


```python
## calculate training error
print 'training error= %.4f' % (1 - np.mean(y_pred == y))
```


```python
print y_pred[:10]
```

可以看看 y_pred 的结果同样是两类，那我们要怎么看每个样本预测的概率呢？可以使用 predict_proba 函数


```python
y_predProbs = logit.predict_proba(X)
```


```python
y_predProbs[:10]
```

看前10个样本的预测结果，左边一列代表分类结果为 Down 的概率，右边一列代表分类结果为 Up 的概率。
不过看这个举棋不定的样子，大概就跟瞎猜差不多。

炒股有风险，炒股需谨慎。
拿来练练手还是可以的。

我们也可以用 statsmodel 中的函数来做逻辑回归的同时筛选特征，做法有点类似于前面的线性回归。但是我们这里用更 formula 的方式来表示，这样看起来更科学一些。
首先导入statsmodels.formula.api


```python
import statsmodels.formula.api as smf
```

然后用glm函数来计算。glm=Generalize Linear Model 广义线性模型。当family=Binomial的时候就是用逻辑回归做二元分类。在 formula 部分我们指定需要预测的 column name 和 成为特征的 column name, 表现为线性求和的方式。对应我们的线性模型。在 data 的部分制定数据集就可以了。是不是很方便！


```python
logit_sm = smf.glm(formula='Direction~Lag1+Lag2+Lag3+Lag4+Lag5+Volume', data=smarket,
                  family=sm.families.Binomial())
```


```python
results = logit_sm.fit()
```


```python
results.summary()
```


```python
y_pred = results.predict()
```


```python
y_pred_class = np.array(['Down']*len(y))
y_pred_class[y_pred < 0.5] = 'Up'
```


```python
print 'training error=%.4f' % (1 - np.mean(y_pred_class == y))
```

这个错误率，也跟瞎猜差不多。但是我们注意到在Lag1和Lag2这两个特征的P值相对于其它特征来说还是比较小的，假如我们把别的都踢掉会怎么样呢？


```python
logit_sm_less = smf.glm(formula='Direction~Lag1+Lag2', data=smarket,
                  family=sm.families.Binomial())
```


```python
results_less = logit_sm_less.fit()
```


```python
results_less.summary()
```


```python
y_pred = results_less.predict()
y_pred_class = np.array(['Down']*len(y))
y_pred_class[y_pred < 0.5] = 'Up'
print 'training error=%.4f' % (1 - np.mean(y_pred_class == y))
```

这个结果看起来也就好了一点点点吧……

练习：以2005年之前的数据为训练集，2005年的数据为测试集，计算logistic regression在测试集上的testing error
