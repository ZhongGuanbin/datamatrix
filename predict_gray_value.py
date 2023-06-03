import numpy as np
import statsmodels.api as sm
from sklearn.neighbors import KernelDensity

'''
1.  对于每个网格区域分割迭代后的灰度分布，根据数据特点，整体来看数据是收缩的，因此可以使用局部加权回归LWR进行预测。
    因为正常情况下，灰度分布的数据是收缩的，因此距离越近的数据点的可信度越高，即权重越大。
    因此我们的数据模型的特点应该是对距离较近的数据点赋予更高的权重，对距离较远的数据点赋予较低的权重。
    此特征正好符合局部加权回归LWR的特点，因此可以使用局部加权回归LWR进行预测。
    将通过局部加权回归LWR预测到的值记为predict_gray_value_lwr,并对其进行二值化，1表示黑色，0表示白色。

2.  通过LWR算法预测的灰度值总体上是准确的，但是当噪点出现在网格中心时，将会产生较大的误差，因此我们需要对网格整体的灰度分布进行分析。
    通过分析网格整体的灰度分布，预测出一个基于整体样品数据推测出的最有可能的灰度值，并将其二值化后与predict_gray_value_lwr作比较。
    由于噪声的出现，数据的分布会变得毫无规律，因此提出采用核密度估计函数KDE算法对网格整体的灰度分布进行分析。
    因为KDE算法不需要事先对概率密度函数进行假设，不限制概率密度函数的形状，可以处理没有明显分布规律的数据。
    并且核密度估计函数KDE算法利用样本数据的密度来近似估计未知数据的密度分布，对于一些无法用已知概率分布来描述的数据，KDE算法具有很大的优势。
    所以我们采用核密度估计函数KDE算法对网格整体的灰度分布进行分析，得到一个基于整体样品数据推测出的最有可能的灰度值predict_gray_value_kde。
    
3.  将predict_gray_value_kde二值化后与同样二值化后的predict_gray_value_lwr作比较，
    如果predict_gray_value_kde与predict_gray_value_lwr相同，则认为预测值是准确的，将predict_gray_value_lwr添加到predict_grey_list。  
    如果predict_gray_value_kde与predict_gray_value_lwr不同，则认为预测值是不准确的，网格存在较大的噪声。
    标准差是方差的平方根，用来表示数据的离散程度。在判断变化趋势时，标准差可以反映区域内的灰度变化的波动程度。
    因此我们可以通过标准差找到网格灰度分布中相对于平均值的偏离程度最大的点，即为噪声点。
    具体做法如下：
        （1）计算网格灰度分布的平均值和标准差，并对网格灰度分布进行标准化处理。
            gray_value_mean = np.mean(gray_tend_list)
            gray_value_std = np.std(gray_tend_list)
            gray_value_scores = (gray_tend_list - gray_value_mean) / gray_value_std
        （2）标准化后的值表示数据点相对于整个网格灰度分布的位置，也称为标准分数。
            标准分数表示的是数据点相对于整个数据集的相对位置，它的值可以用来判断数据点是否偏离了数据集的中心位置，
            将数据点的值转换为该值与整个数据集平均值的差值除以数据集的标准差，即该数据点在数据集中相对于平均值的偏离程度。
        （3）剔除标准差最大的数据点，重新计算predict_gray_value_kde和predict_gray_value_lwr。
            max_index = np.argmax(np.abs(gray_value_scores))
            gray_tend_list = np.delete(gray_tend_list, max_index)
        （4）再次比较predict_gray_value_kde和predict_gray_value_lwr，如果不同，则重复上述步骤：
            重新计算数据集的平均值和标准差，剔除标准差最大的数据点。
                gray_value_mean = np.mean(gray_tend_list)
                gray_value_std = np.std(gray_tend_list)
                gray_value_scores = (gray_tend_list - gray_value_mean) / gray_value_std
                max_index = np.argmax(np.abs(gray_value_scores))
                gray_tend_list = np.delete(gray_tend_list, max_index)
        （5）直到predict_gray_value_kde和predict_gray_value_lwr相同，或者网格灰度分布中的数据点个数小于等于2为止。
        （6）如果网格灰度分布中的数据点个数小于等于2都无法判出，则认为网格中的噪声点过多，无法进行预测。
'''


def predict_by_lwr(value_list):
    # 将灰度分布转换为数组
    x = np.arange(len(value_list))
    y = np.array(value_list)

    # 进行局部加权回归LWR预测，得到二值化后的预测值predict_gray_value_lwr
    model = sm.nonparametric.KernelReg(endog=y, exog=x, var_type='c')
    y_pred, _ = model.fit(np.array([[x[-1] + 1]]))

    # 计算下一个点的可能值
    predict_gray_value_lwr = y_pred[0]

    # 根据阈值调整预测值,将predict_gray_value_lwr二值化，1表示黑色，0表示白色
    if predict_gray_value_lwr > 128:
        predict_gray_value_lwr = 0
    else:
        predict_gray_value_lwr = 1

    return predict_gray_value_lwr


def predict_by_kde(value_list):
    # 将灰度分布转换为数组
    x = np.arange(len(value_list))
    y = np.array(value_list)

    # 通过KDE算法对网格整体的灰度分布进行分析，得到一个基于整体样品数据推测出的最有可能的灰度值predict_gray_value_kde
    # 创建KernelDensity类的实例
    kde = KernelDensity(bandwidth="auto", kernel='gaussian')

    # 通过fit()方法拟合数据,自动选择带宽
    kde.fit(y.reshape(-1, 1))

    # 使用score_samples()方法进行预估
    x_eval = np.linspace(0, max(value_list), 1000).reshape(-1, 1)
    y_eval = np.exp(kde.score_samples(x_eval))

    # 找出y_eval最大值所在的索引
    max_index = np.argmax(y_eval)

    # 最有可能的真实值即为对应的x_eval值
    predict_gray_value_kde = x_eval[max_index][0]

    # 将predict_gray_value_kde二值化，1表示黑色，0表示白色
    if predict_gray_value_kde > 128:
        predict_gray_value_kde = 0
    else:
        predict_gray_value_kde = 1

    return predict_gray_value_kde


# 定义一个预测灰度值的函数predict_gray_value
def predict_gray_value(grey_tend_list_total):
    # 存储根据灰度分布得到的预测灰度值
    predict_grey_list = []

    # 循环遍历每个灰度分布
    for gray_tend_list in grey_tend_list_total:
        while True:
            # 通过LWR获取预测值
            predict_gray_value_lwr = predict_by_lwr(gray_tend_list)

            # 通过KDE获取预测值
            predict_gray_value_kde = predict_by_kde(gray_tend_list)

            # 如果predict_gray_value_kde与predict_gray_value_lwr不同，则认为预测值是不准确的，网格存在较大的噪声。
            if predict_gray_value_lwr != predict_gray_value_kde:
                # 计算网格灰度分布的平均值和标准差，对网格灰度分布进行标准化处理，剔除标准差最大的数据点
                gray_value_mean = np.mean(gray_tend_list)
                gray_value_std = np.std(gray_tend_list)
                gray_value_scores = (gray_tend_list - gray_value_mean) / gray_value_std
                max_index = np.argmax(np.abs(gray_value_scores))
                gray_tend_list = np.delete(gray_tend_list, max_index)

                # 如果网格灰度分布中的数据点个数小于等于2，则认为网格中的噪声点过多，无法进行预测。
                if len(gray_tend_list) <= 2:
                    predict_grey_list.append(predict_gray_value_lwr)
                    break
            else:
                # 预测值相同，将predict_gray_value_lwr添加到predict_grey_list中，并结束循环
                predict_grey_list.append(predict_gray_value_lwr)
                break

    return predict_grey_list
