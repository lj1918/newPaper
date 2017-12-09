# -*- coding: utf-8 -*-
"""
scikit-learn PCA类介绍:
在scikit-learn中，与PCA相关的类都在sklearn.decomposition包中。
最常用的PCA类就是sklearn.decomposition.PCA，我们下面主要也会讲解基于这个类的使用的方法。
   除了PCA类以外，最常用的PCA相关类还有KernelPCA类，在原理篇我们也讲到了，
它主要用于非线性数据的降维，需要用到核技巧。因此在使用的时候需要选择合适的核函数并对核函数的参数进行调参。

　　另外一个常用的PCA相关类是IncrementalPCA类，它主要是为了解决单机内存限制的。有时候我们的样本量可能是上百万+，
维度可能也是上千，直接去拟合数据可能会让内存爆掉， 此时我们可以用IncrementalPCA类来解决这个问题。
IncrementalPCA先将数据分成多个batch，然后对每个batch依次递增调用partial_fit函数，这样一步步的得到最终的样本最优降维。

　 此外还有SparsePCA和MiniBatchSparsePCA。他们和上面讲到的PCA类的区别主要是使用了L1的正则化，
这样可以将很多非主要成分的影响度降为0，这样在PCA降维的时候我们仅仅需要对那些相对比较主要的成分进行PCA降维，
避免了一些噪声之类的因素对我们PCA降维的影响。SparsePCA和MiniBatchSparsePCA之间的区别则是
MiniBatchSparsePCA通过使用一部分样本特征和给定的迭代次数来进行PCA降维，
以解决在大样本时特征分解过慢的问题，当然，代价就是PCA降维的精确度可能会降低。
使用SparsePCA和MiniBatchSparsePCA需要对L1正则化参数进行调参。

1）n_components：这个参数可以帮我们指定希望PCA降维后的特征维度数目。最常用的做法是直接指定降维到的维度数目，
此时n_components是一个大于等于1的整数。当然，我们也可以指定主成分的方差和所占的最小比例阈值，
让PCA类自己去根据样本特征方差来决定降维到的维度数，此时n_components是一个（0，1]之间的数。
当然，我们还可以将参数设置为"mle", 此时PCA类会用MLE算法根据特征的方差分布情况自己去选择一定数量的主成分特征来降维。
我们也可以用默认值，即不输入n_components，此时n_components=min(样本数，特征数)。

2）whiten ：判断是否进行白化。所谓白化，就是对降维后的数据的每个特征进行归一化，让方差都为1.对于PCA降维本身来说，
一般不需要白化。如果你PCA降维后有后续的数据处理动作，可以考虑白化。默认值是False，即不进行白化。

3）svd_solver：即指定奇异值分解SVD的方法，由于特征分解是奇异值分解SVD的一个特例，一般的PCA库都是基于SVD实现的。
有4个可以选择的值：{‘auto’, ‘full’, ‘arpack’, ‘randomized’}。randomized一般适用于数据量大，
数据维度多同时主成分数目比例又较低的PCA降维，它使用了一些加快SVD的随机算法。 full则是传统意义上的SVD，
使用了scipy库对应的实现。arpack和randomized的适用场景类似，区别是randomized使用的是scikit-learn自己的SVD实现，
而arpack直接使用了scipy库的sparse SVD实现。默认是auto，即PCA类会自己去在前面讲到的三种算法里面去权衡，
选择一个合适的SVD算法来降维。一般来说，使用默认值就够了。

除了这些输入参数外，有两个PCA类的成员值得关注。第一个是explained_variance_，它代表降维后的各主成分的方差值。
方差值越大，则说明越是重要的主成分。第二个是explained_variance_ratio_，它代表降维后的各主成分的方差值占总方差值的比例，
这个比例越大，则越是重要的主成分。

@author: 刘军

基本上用PCA和KernelPCA即可
"""

from sklearn.decomposition import PCA,IncrementalPCA,KernelPCA,FastICA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis #  LDA
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics  import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
from Load_data import *

raw_x,y = load_zhiwai_data2()

n_components = 10

#================== PCA 识别准确率0.882352941176
#pca = PCA(n_components=n_components,whiten=True)
#x = pca.fit_transform(raw_x)

#============== KernelPCA 识别准确率0.882352941176
#kpca = KernelPCA(n_components=n_components)
#x = kpca.fit_transform(raw_x)

#========================FastICA 识别准确率0.176470588235 ？？
#ica = FastICA(n_components=n_components)
#x = ica.fit_transform(raw_x,y) 

'''
另外一个常用的PCA相关类是IncrementalPCA类，它主要是为了解决单机内存限制的。
有时候我们的样本量可能是上百万+，维度可能也是上千，直接去拟合数据可能会让内存爆掉， 
此时我们可以用IncrementalPCA类来解决这个问题。IncrementalPCA先将数据分成多个batch，
然后对每个batch依次递增调用partial_fit函数，这样一步步的得到最终的样本最优降维
'''
#================= IncrementalPCA 识别准确率0.647058823529
#ipca = IncrementalPCA(n_components=n_components,batch_size=4)
#x = ipca.fit_transform(raw_x) 

#================= LDA 识别准确率0.823529411765
lda = LinearDiscriminantAnalysis(n_components=n_components, tol=0.0001)
x = lda.fit_transform(raw_x,y)


train_x,test_x,train_y,test_y = train_test_split(x,y,train_size=0.7,random_state=0)

model = SVC()
model.fit(train_x,train_y)
predict_test_y =  model.predict(test_x)
print(accuracy_score(test_y,predict_test_y))