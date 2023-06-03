import os.path
import cv2
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.neighbors import KernelDensity
import pylibdmtx.pylibdmtx as dmtx
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt


# 定义寻峰函数，寻找可能的分割线
def find_peak(y_input):
    # 去基线，y＜7000的认为是噪音，扣除
    # 7000的取值原因：最大尺寸26*26，所以1格灰度值1000/26*255
    # 至少有1格模式边，就算数据区某行或者某列全为0，再加上部分变形，至少7000
    y_sub = np.where(y_input < 7000, 0, y_input)

    # 对y进行差分
    y_diff = np.diff(y_sub)

    # 对y_diff求符号
    y_diff_sign = np.sign(y_diff)

    # 对y_diff_sign再求一次差分
    y_diff_sign_diff = np.diff(y_diff_sign)

    # 遍历y_diff_sign_diff，
    # 将其中y_diff_sign_diff[i]=-2(单峰)或y_diff_sign_diff[i] == -1（梯形峰，有多个相等的最大值）的i值变为i+1后存储到数组peak_list_origin中
    peak_list_origin = []
    for i in range(len(y_diff_sign_diff)):
        if y_diff_sign_diff[i] == -2 or y_diff_sign_diff[i] == -1:
            peak_list_origin.append(i + 1)

    # 对peak_list_origin做差分，查看两个峰值之间的间隔，保存在数组peak_list_diff中
    peak_list_diff_origin = np.diff(peak_list_origin)

    # 由于初步的峰存在很多噪声，所以会有很多很小的间隔值。
    # 对peak_list_diff排序，则排在前面很小的值为噪声的可能性很大，越往后越有可能是正确的峰。
    # 因此为了预测出真实的峰间距，需要更多的参考后面的数据，即对排序后的peak_list_diff的数据，越往后可信度越高。
    # 因此采用指数加权移动平均模型（EWMA）来预测真实的峰间距

    # 对peak_list_diff_origin排序
    peak_list_diff_origin.sort()

    # 将peak_list_diff_origin数据转换为pandas序列
    peak_list_diff = pd.Series(peak_list_diff_origin)

    # 计算指数加权移动平均
    peak_list_diff_ewma = peak_list_diff.ewm(alpha=0.3).mean()

    # 得到预测间距值
    prediction_peak_spacing = peak_list_diff_ewma.iloc[-1]

    # 原始的峰列表由于噪音问题会存在很多肩峰和重峰的现象。因此我们需要判断哪些峰是同一组的。
    # 将肩峰重峰放在一个数组内，再从这个数组内找出一个峰来代表真实的峰。
    # 创建一个数组peak_list_group用来存放分好组以后的峰，创建一个数组temp_list用来分组时存放同一组的峰
    peak_list_find_peak = peak_list_origin.copy()
    peak_list_group = []
    temp_list = [peak_list_find_peak[0]]

    # 遍历peak_list_find_peak
    for i in range(len(peak_list_find_peak) - 1):

        # 差分，得到峰间距，根据峰间距来判断是否是同一组的峰
        x_diff = peak_list_find_peak[i + 1] - peak_list_find_peak[i]

        # 因为单个二维码水平或者竖直方向最多有26个模块，图像在1000*1000的空间内，所以正常情况下的最小间距是1000/26大约40，
        # 再结合我们预测的真实峰间距，若峰间距小于max（8，prediction_peak_spacing/2），则认为是同一组的峰，
        # 直到峰间距大于max（8，prediction_peak_spacing/2）
        if x_diff <= max(8, prediction_peak_spacing / 2):
            temp_list.append(peak_list_find_peak[i + 1])
        else:
            peak_list_group.append(temp_list)
            temp_list = [peak_list_find_peak[i + 1]]

    # 如果是最后一组峰（肩峰或单峰），则直接将其添加到peak_list_group中，避免漏掉尾峰
    if peak_list_find_peak[-1] - temp_list[-1] == 0:
        peak_list_group.append(temp_list)

    # 找出每组峰里最有代表性的峰作为该组峰真正的峰存储到peak_list_group_max，这里采取选择强度最大的峰作为真正的峰。
    peak_list_group_max = []

    # 遍历peak_list_group
    for i in range(len(peak_list_group)):

        # 如果只有一个峰，则直接将其添加到peak_list_group_max中
        if len(peak_list_group[i]) == 1:
            peak_list_group_max.append(int(peak_list_group[i][0]))

        # 如果有多个峰，则找出最大值所对应的j值
        else:
            peak_y_list = []
            for j in range(len(peak_list_group[i])):
                peak_y_list.append(y_sub[int(peak_list_group[i][j])])
            max_y_sub = max(peak_y_list)

            # 如果只有1个max_y_sub，则取最大值所对应的x值
            if peak_y_list.count(max_y_sub) == 1:
                max_y_sub_idx = peak_y_list.index(max_y_sub)
                peak_list_group_max.append(int(peak_list_group[i][max_y_sub_idx]))

            # 如果出现了多个峰的y_sub值相同的情况，则取多个y_sub对应的x值的平均值
            else:
                max_y_sub_idx = [i for i, x in enumerate(peak_y_list) if x == max_y_sub]
                temp = 0
                for j in range(len(max_y_sub_idx)):
                    temp += peak_list_group[i][max_y_sub_idx[j]]
                peak_list_group_max.append(int(temp / len(max_y_sub_idx)))

    # 遍历peak_list_group_max,进行补峰
    # 补峰后，peak_list_group_max长度改变，要注意更新peak_list_group_max长度
    peak_list_len = len(peak_list_group_max)
    i = 0
    while i < peak_list_len - 1:

        # 计算峰间距
        peak_diff = peak_list_group_max[i + 1] - peak_list_group_max[i]

        # 插入峰的个数统计值
        insert_peak_num = 1

        # 如果峰间距大于2倍预测峰间距，则需要插入新的峰
        while peak_diff > 2 * prediction_peak_spacing:
            # 插入值.由数据特征可知，由于存在大量噪音，导致预测的峰间距实际上会偏小，因此在插入峰间距时给予一定的权重进行补偿
            insert_value = int(peak_list_group_max[i] + prediction_peak_spacing * 1.2 * insert_peak_num)

            # 在第一个峰间隔预测峰间距处插入一个峰
            peak_list_group_max.insert(i + insert_peak_num, insert_value)

            # 重新计算峰间距
            peak_diff = peak_list_group_max[i + 1 + insert_peak_num] - peak_list_group_max[i + insert_peak_num]

            # 插入峰的个数统计值加1
            insert_peak_num += 1

            # 更新列表长度
            peak_list_len += 1

        i += 1

    return peak_list_group_max


# 定义网格划分函数，得到各网格的四顶点坐标
def grid_division(peak_list_x, peak_list_y, is_point=False):
    # 存储列信息，即每列的左右端点坐标
    column_list = []

    # 遍历peak_list_x
    for i in range(len(peak_list_x) - 1):
        # 暂存每列的左右端点坐标
        temp_list = [peak_list_x[i], peak_list_x[i + 1]]

        # 将暂存的每列的左右端点坐标添加到column_list中
        column_list.append(temp_list)

    # 存储行信息，即每行的上下端点坐标
    row_list = []

    # 遍历peak_list_y
    for i in range(len(peak_list_y) - 1):
        # 暂存每行的上下端点坐标
        temp_list = [peak_list_y[i], peak_list_y[i + 1]]

        # 将暂存的每行的上下端点坐标添加到row_list中
        row_list.append(temp_list)

    # 如果is_point为True，则说明是点状二维码
    # 点状二维码只取奇数行和奇数列，删除column_list和row_list的偶数行和偶数列
    if is_point:
        # 删除column_list的偶数列, 即索引为奇数的列
        column_list = column_list[::2]

        # 删除row_list的偶数行, 即索引为奇数的行
        row_list = row_list[::2]

    # 定义矩阵grid_points，存储网格划分后各网格的四顶点坐标
    grid_points = []

    # 遍历行
    for i in range(len(row_list)):
        # 遍历列
        for j in range(len(column_list)):
            # 计算每个网格的四个顶点坐标
            # 左上角顶点坐标
            left_top = (column_list[j][0], row_list[i][0])

            # 右上角顶点坐标
            right_top = (column_list[j][1], row_list[i][0])

            # 左下角顶点坐标
            left_bottom = (column_list[j][0], row_list[i][1])

            # 右下角顶点坐标
            right_bottom = (column_list[j][1], row_list[i][1])

            # 将每个网格的四个顶点坐标存储到grid_points中
            grid_points.append([left_top, right_top, left_bottom, right_bottom])

    # 返回网格划分后各网格的四顶点坐标
    return grid_points


# 定义网格迭代分割函数，返回分割结果——一个存储着每一层单位化灰度值的列表
# 若传入的是网格列表，则返回一组单位化灰度值的列表
def grid_iterative_segmentation(grid_points_list_input, binary_image_input):
    # 定义存储一组单位化灰度值列表的数组
    gray_value_list_array = []

    # 遍历网格列表，对每一个网格进行迭代分割
    for grid_points in grid_points_list_input:
        # 定义存储每一层单位化灰度值的列表
        gray_value_list = []

        # 获取网格的四个顶点坐标，可以获得该网格的长和宽
        grid_length = grid_points[1][0] - grid_points[0][0]
        grid_width = grid_points[2][1] - grid_points[0][1]

        # 迭代分割网格，直到网格的长grid_length<=2或者宽grid_width<=2
        while grid_length > 2 and grid_width > 2:
            # 首先计算当前四个顶点连成的四边形的单位化灰度值
            # 即累加四个顶点连成的四边形内每个像素点的灰度值，然后除以四边形面积
            # 然后将该单位化灰度值存储到gray_value_list中
            mask = np.zeros_like(binary_image_input)
            cv2.fillPoly(mask, [np.array([grid_points])], 255)
            gray_value = np.mean(binary_image_input[mask > 0])
            gray_value_list.append(gray_value)

            # 再然后将网格的长和宽都-2，得到新的网格，计算新的网格的单位化灰度值，添加到gray_value_list中
            # 重复上述过程，直到网格的长grid_length<=2或者宽grid_width<=2，
            # 返回gray_value_list

            # 更新网格的长和宽
            grid_length -= 2
            grid_width -= 2

            # 重新计算网格的四个顶点坐标
            grid_points = [(grid_points[0][0] + 1, grid_points[0][1] + 1),
                           (grid_points[1][0] - 1, grid_points[1][1] + 1),
                           (grid_points[2][0] + 1, grid_points[2][1] - 1),
                           (grid_points[3][0] - 1, grid_points[3][1] - 1)]

        # 将gray_value_list添加到gray_value_list_array中
        gray_value_list_array.append(gray_value_list)

    # 返回gray_value_list_array
    return gray_value_list_array


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
    model = sm.nonparametric.KernelReg(endog=y, exog=x, var_type='o')
    y_pred, _ = model.fit(np.array([[x[-1] + 1]]))

    # 计算下一个点的可能值
    predict_gray_value_lwr = y_pred[0]

    # 根据阈值调整预测值,将predict_gray_value_lwr二值化，0表示黑色，255表示白色
    if predict_gray_value_lwr > 128:
        predict_gray_value_lwr = 255
    else:
        predict_gray_value_lwr = 0

    return predict_gray_value_lwr


def predict_by_kde(value_list):
    # 将灰度分布转换为数组
    y = np.array(value_list)

    # 通过KDE算法对网格整体的灰度分布进行分析，得到一个基于整体样品数据推测出的最有可能的灰度值predict_gray_value_kde
    # 创建KernelDensity类的实例
    kde = KernelDensity(bandwidth="scott", kernel='gaussian')

    # 通过fit()方法拟合数据,自动选择带宽
    kde.fit(y.reshape(-1, 1))

    # 使用score_samples()方法进行预估
    x_eval = np.linspace(0, max(value_list), 5000).reshape(-1, 1)
    y_eval = np.exp(kde.score_samples(x_eval))

    # 找出y_eval最大值所在的索引
    max_index = np.argmax(y_eval)

    # 最有可能的真实值即为对应的x_eval值
    predict_gray_value_kde = x_eval[max_index][0]

    # 将predict_gray_value_kde二值化，0表示黑色，255表示白色
    if predict_gray_value_kde > 128:
        predict_gray_value_kde = 255
    else:
        predict_gray_value_kde = 0

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
                    predict_grey_list.append(predict_gray_value_kde)
                    break
            else:
                # 预测值相同，将predict_gray_value_lwr添加到predict_grey_list中，并结束循环
                predict_grey_list.append(predict_gray_value_kde)
                break

    return predict_grey_list


# 定义解码函数，输入图片的路径，输出解码后的数据矩阵
def decode_dmtx(image_path):
    # 采用cv2读取路径为image_path的PNG图片，将其灰度化
    image = cv2.imread(image_path)

    # # 使用中值滤波对图像进行去噪
    # median_image = cv2.medianBlur(image, 5)

    # # 采用cv2的双边滤波方法，去噪,去除高斯白噪声，尽可能保留边缘信息
    # denoised_image = cv2.bilateralFilter(image, 9, 75, 75)

    # # 光照均匀后的图像
    # adaptive_light_image = adaptive_light_correction(bgr_image)
    # plt.imshow(adaptive_light_image)
    # plt.show()
    #
    # adaptive_light_image = cv2.cvtColor(adaptive_light_image, cv2.COLOR_BGR2GRAY)
    # plt.imshow(adaptive_light_image)
    # plt.show()


    # # 使用retinex均匀光照
    # retinex_image = apply_retinex(bgr_image)
    # plt.imshow(retinex_image)
    # plt.show()
    # retinex_image = cv2.cvtColor(retinex_image, cv2.COLOR_BGR2GRAY)

    # CLAHE对比度增强
    clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(5, 5))
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img = clahe.apply(img)
    # plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # plt.show()

    # # 中值滤波器去噪
    # img_median = cv2.medianBlur(img, 3)
    # plt.imshow(cv2.cvtColor(img_median, cv2.COLOR_BGR2RGB))
    # plt.title('median')
    # plt.show()

    # # 对灰度图像进行泊松噪音去除
    # img_poisson = cv2.fastNlMeansDenoising(img_median, None, h=9, templateWindowSize=7, searchWindowSize=21)
    # plt.imshow(cv2.cvtColor(img_poisson, cv2.COLOR_BGR2RGB))
    # plt.title('poisson')
    # plt.show()

    # 采用cv2的双边滤波方法，去噪,去除高斯白噪声，尽可能保留边缘信息
    img_bilateral = cv2.bilateralFilter(img, 9, 75, 75)
    resized_bilateral_image = cv2.resize(img_bilateral, (1000, 1000), interpolation=cv2.INTER_LINEAR)
    # plt.title('bilateral')
    # plt.show()

    # # 采用cv2的双边滤波方法，去噪,去除高斯白噪声，尽可能保留边缘信息
    # denoised_image = cv2.bilateralFilter(img, 9, 75, 75)

    # # 对灰度图像进行Laplacian滤波
    # denoised_image = cv2.Laplacian(image, cv2.CV_64F)
    #
    # # 将滤波后的图像进行归一化和类型转换
    # denoised_image = cv2.convertScaleAbs(denoised_image)

    # # 对灰度图像进行Sobel滤波
    # sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    # sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    # sobel = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    # sobel = cv2.convertScaleAbs(sobel)

    # denoised_image = cv2.GaussianBlur(image, (13, 13), 0)

    # 采用cv2进行二值化，二值化的阈值用OTSU算法自动计算
    ret, binary_image = cv2.threshold(img_bilateral, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 应用局部自适应阈值滤波
    # binary_image = cv2.adaptiveThreshold(img_bilateral, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 111, 2)

    # 缩放到1000*1000的像素空间中,增大空间是为了区域迭代分割有更多的数据点，能做更准确的分析
    resized_binary_image = cv2.resize(binary_image, (1000, 1000), interpolation=cv2.INTER_LINEAR)

    # 计算每个像素点与右侧一个像素点的灰度差值
    gx = np.abs(np.diff(np.asarray(resized_binary_image), axis=1))

    # 统计每一列像素的差值和，最右侧的像素点则与0做差值
    col_sum = np.sum(gx, axis=0)
    col_sum = np.append(col_sum, 0)

    # 水平方向横纵坐标数组
    y_h = col_sum

    # 计算水平方向分割线
    find_peak_list_x = find_peak(y_h)

    # 统计每一行像素的差值和，最下方的像素点则与0做差值
    gy = np.abs(np.diff(np.asarray(resized_binary_image), axis=0))
    row_sum = np.sum(gy, axis=1)
    row_sum = np.append(row_sum, 0)

    # 垂直方向横纵坐标数组
    y_v = row_sum

    # 计算垂直方向分割线
    find_peak_list_y = find_peak(y_v)

    # 绘制图像和分割线
    plt.imshow(resized_bilateral_image, cmap='gray')
    for x in find_peak_list_x:
        plt.axvline(x=x, color='r')
    for y in find_peak_list_y:
        plt.axhline(y=y, color='r')

    # 展示图像
    plt.show()

    # 得到网格划分后每个网格的四个顶点坐标
    grid_points_list = grid_division(peak_list_x=find_peak_list_x, peak_list_y=find_peak_list_y, is_point=False)

    # 得到网格列表内每个网格的灰度分布趋势
    grid_grey_tend_list = grid_iterative_segmentation(grid_points_list_input=grid_points_list,
                                                      binary_image_input=resized_bilateral_image)
    print(grid_grey_tend_list)
    # 得到每个网格二值化后的值，1表示黑色，0表示白色
    predict_grey_value_list = predict_gray_value(grid_grey_tend_list)

    # 返回解码后的数据矩阵
    return predict_grey_value_list


# 光照均匀函数
def adaptive_light_correction(img):
    height = img.shape[0]
    width = img.shape[1]

    HSV_img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    V = HSV_img[:,:,2]

    kernel_size = min(height, width)
    if kernel_size % 2 == 0:
        kernel_size -= 1

    SIGMA1 = 15
    SIGMA2 = 80
    SIGMA3 = 250
    q = np.sqrt(2.0)
    F = np.zeros((height,width,3),dtype=np.float64)
    F[:,:,0] = cv2.GaussianBlur(V,(kernel_size, kernel_size),SIGMA1 / q)
    F[:,:,1] = cv2.GaussianBlur(V,(kernel_size, kernel_size),SIGMA2 / q)
    F[:,:,2] = cv2.GaussianBlur(V,(kernel_size, kernel_size),SIGMA3 / q)
    F_mean = np.mean(F,axis=2)
    average = np.mean(F_mean)
    gamma = np.power(0.5,np.divide(np.subtract(average,F_mean),average))
    out = np.power(V/255.0,gamma)*255.0
    HSV_img[:,:,2] = out
    img = cv2.cvtColor(HSV_img,cv2.COLOR_HSV2BGR)
    return img


# retinex均匀光照
def apply_retinex(image):
    # 使用opencv的Retinex算法进行图像均衡化
    img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(img)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    v = clahe.apply(v)
    img = cv2.merge((h, s, v))
    img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
    return img


root = tk.Tk()

root.withdraw()

f_path = filedialog.askopenfilename()

test_data = decode_dmtx(f_path)
print(test_data)

# # 设置灰度图的大小
# length = int(len(test_data) ** 0.5)
#
# # 将灰度值列表转成灰度图像
# predict_grey_value_array = np.array(test_data, dtype=np.uint8)
# predict_grey_image = predict_grey_value_array.reshape(length, length)
#
# # 缩放灰度图
# resized_grey_image = cv2.resize(predict_grey_image, (400, 400), interpolation=cv2.INTER_AREA)
#
# # 在图像四周添加30像素的空白
# resized_grey_image = cv2.copyMakeBorder(resized_grey_image, 50, 50, 50, 50, cv2.BORDER_CONSTANT, value=0)
#
# # 保存路径
# save_path_dir = os.path.join(os.path.dirname(f_path), 'output')
# os.makedirs(save_path_dir, exist_ok=True)
# save_path = os.path.join(save_path_dir, os.path.basename(f_path))
#
# # 保存图片
# cv2.imwrite(save_path, resized_grey_image)
#
# # 解码灰度图
# decoded_data = dmtx.decode(cv2.imread(save_path, cv2.IMREAD_GRAYSCALE))
#
# # 打印解码结果
# print(decoded_data)
