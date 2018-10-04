# one_hot 编码

## 什么是one_hot编码

在很多机器学习任务中，特征值并不总是连续的，而是一些类别。这些类别之间没有序关系，不能简单使用1、2、 3来进行编码。所以需要使用一种表示类别且没有序关系的编码。

## 实现

独热编码即 One-Hot 编码，又称一位有效编码，其方法是使用N位状态寄存器来对N个状态进行编码，每个状态都由他独立的寄存器位，并且在任意时候，其中只有一位有效。

> 自然状态码为：000,001,010,011,100,101
>
>  独热编码为：000001,000010,000100,001000,010000,100000

 可以这样理解，对于每一个特征，如果它有m个可能值，那么经过独热编码后，就变成了m个二元特征。并且，这些特征互斥，每次只有一个激活。因此，数据会变成稀疏的。

## 优点

1. 解决了分类器不好处理属性数据的问题
2. 在一定程度上也起到了扩充特征的作用

## 代码

使用Scikit-learn实现one_hot：

```python
from sklearn import preprocessing
enc = preprocessing.OneHotEncoder()
enc.fit([[0, 0, 3], [1, 1, 0], [0, 2, 1], [1, 0, 2]])
enc.transform([[0, 1, 3]]).toarray()
```

使用tensorflow实现：

```python
import numpy as np
import tensorflow as tf

SIZE=6
CLASS=8
label1=tf.constant([0,1,2,3,4,5,6,7])
sess1=tf.Session()
print('label1:',sess1.run(label1))
b = tf.one_hot(label1,CLASS,1,0)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(b)
    print('after one_hot',sess.run(b))

--------------------
最后的输出为：

label1: [0 1 2 3 4 5 6 7]
after one_hot:
 [[1 0 0 0 0 0 0 0]
 [0 1 0 0 0 0 0 0]
 [0 0 1 0 0 0 0 0]
 [0 0 0 1 0 0 0 0]
 [0 0 0 0 1 0 0 0]
 [0 0 0 0 0 1 0 0]
 [0 0 0 0 0 0 1 0]
 [0 0 0 0 0 0 0 1]]


#本文来自 超屌的温jay 的CSDN 博客 ，全文地址请点击：https://blog.csdn.net/wenqiwenqi123/article/details/78055740?utm_source=copy 
```



来自官网`http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html`

```Python
>>> from sklearn.preprocessing import OneHotEncoder
>>> enc = OneHotEncoder(handle_unknown='ignore')
>>> X = [['Male', 1], ['Female', 3], ['Female', 2]]
>>> enc.fit(X)
... 
OneHotEncoder(categorical_features=None, categories=None,
       dtype=<... 'numpy.float64'>, handle_unknown='ignore',
       n_values=None, sparse=True)
```

```Python
>>> enc.categories_
[array(['Female', 'Male'], dtype=object), array([1, 2, 3], dtype=object)]
>>> enc.transform([['Female', 1], ['Male', 4]]).toarray()
array([[1., 0., 1., 0., 0.],
       [0., 1., 0., 0., 0.]])
>>> enc.inverse_transform([[0, 1, 1, 0, 0], [0, 0, 0, 1, 0]])
array([['Male', 1],
       [None, 2]], dtype=object)
>>> enc.get_feature_names()
array(['x0_Female', 'x0_Male', 'x1_1', 'x1_2', 'x1_3'], dtype=object)
```

