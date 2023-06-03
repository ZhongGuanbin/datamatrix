import os.path
import tkinter as tk
from tkinter import filedialog

import cv2
import numpy as np
import pandas as pd
import pylibdmtx.pylibdmtx as dmtx
import statsmodels.api as sm


# 定义寻峰函数，寻找可能的分割线
def find_peak(y_input):
    # 去基线
    y_sub = np.where(y_input < 2400, 0, y_input)

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
    peak_list_diff_ewma = peak_list_diff.ewm(alpha=0.355).mean()

    # 得到预测间距值
    prediction_peak_spacing = peak_list_diff_ewma.iloc[-1]

    # 原始的峰列表由于噪音问题会存在很多肩峰和裂峰的现象。因此我们需要判断哪些峰是同一组的。
    # 将肩峰裂峰放在一个数组内，再从这个数组内找出一个峰来代表真实的峰。
    # 创建一个数组peak_list_group用来存放分好组以后的峰，创建一个数组temp_list用来分组时存放同一组的峰
    peak_list_find_peak = peak_list_origin.copy()
    peak_list_group = []
    temp_list = [peak_list_find_peak[0]]

    # 遍历peak_list_find_peak
    for i in range(len(peak_list_find_peak) - 1):

        # 差分，得到峰间距，根据峰间距来判断是否是同一组的峰
        x_diff = peak_list_find_peak[i + 1] - peak_list_find_peak[i]

        # 因为单个二维码水平或者竖直方向最多有26个模块，图像在256*256的空间内，所以正常情况下的最小间距是256/26大约10，
        # 再结合我们预测的真实峰间距，若峰间距小于max（3，prediction_peak_spacing / 2），则认为是同一组的峰，
        # 直到峰间距大于max（3，prediction_peak_spacing / 2）
        if x_diff <= max(3, prediction_peak_spacing / 2):
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

            # 如果出现了多个峰的y_sub值相同的情况，则取最大y_sub对应的x值
            else:
                max_y_sub_idx = [i for i, x in enumerate(peak_y_list) if x == max_y_sub]
                peak_list_group_max.append(int(peak_list_group[i][max_y_sub_idx[0]]))

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


# 定义网格迭代分割函数，返回分割结果——一个存储着每一层单位面积灰度值的列表
# 若传入的是网格列表，则返回一组单位面积灰度值的列表
def grid_iterative_segmentation(grid_points_list_input, binary_image_input):
    # 定义存储一组单位面积灰度值列表的数组
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
            # 首先计算当前四个顶点连成的四边形的单位面积灰度值
            # 即累加四个顶点连成的四边形内每个像素点的灰度值，然后除以四边形面积
            # 然后将该单位化灰度值存储到gray_value_list中
            mask = np.zeros_like(binary_image_input)
            cv2.fillPoly(mask, [np.array([grid_points])], 255)
            gray_value = np.mean(binary_image_input[mask > 0])
            gray_value_list.append(gray_value)

            # 再然后将网格的长和宽都-2，得到新的网格，计算新的网格的单位面积灰度值，添加到gray_value_list中
            # 重复上述过程，直到网格的长grid_length<=2或者宽grid_width<=2，
            # 返回gray_value_list

            # 更新网格的长和宽
            # 如果网格的长和宽都大于2，则长和宽都-2
            if grid_length > 2 and grid_width > 2:
                grid_length -= 2
                grid_width -= 2
            # 如果网格得长小于等于2，但是宽大于2，则宽-2
            elif grid_length <= 2 < grid_width:
                grid_width -= 2
            # 如果网格得宽小于等于2，但是长大于2，则长-2
            else:
                grid_length -= 2

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


# 定义一个预测灰度值的函数predict_gray_value
def predict_gray_value(grey_tend_list_total):
    # 存储根据灰度分布得到的预测灰度值
    predict_grey_list = []

    # 循环遍历每个灰度分布
    for gray_tend_list in grey_tend_list_total:
        # 将灰度分布转换为数组
        x = np.arange(len(gray_tend_list))
        y = np.array(gray_tend_list)

        # 进行局部加权回归LWR预测，得到二值化后的预测值predict_gray_value_lwr
        model = sm.nonparametric.KernelReg(endog=y, exog=x, var_type='o')
        y_fit, _ = model.fit(x)

        # 计算预测点
        x_pred = np.arange(len(gray_tend_list), len(gray_tend_list) + 1)
        y_pred, _ = model.fit(x_pred)
        predict_gray_value_lwr = y_pred[0]

        # 根据阈值调整预测值,将predict_gray_value_lwr二值化，1表示黑色，0表示白色
        if predict_gray_value_lwr > 128:
            predict_gray_value_lwr = 255
        else:
            predict_gray_value_lwr = 0

        # 将预测值添加到predict_grey_list中
        predict_grey_list.append(predict_gray_value_lwr)

    return predict_grey_list


# 定义解码函数，输入图片的路径，输出解码后的数据矩阵
def decode_dmtx(image_path):
    # 采用cv2读取路径为image_path的PNG图片
    image = cv2.imread(image_path)

    # 放缩到256*256的像素空间中
    image = cv2.resize(image, (256, 256))

    # CLAHE对比度增强
    clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(5, 5))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = clahe.apply(image)

    # 采用cv2的双边滤波方法，去噪,尽可能保留边缘信息
    # 初始化带宽k=5，带宽越小，边缘信息保存得越好
    denoised_bw = 5
    denoised_image = cv2.bilateralFilter(image, denoised_bw, 75, 75)

    # 采用cv2进行二值化，二值化的阈值用OTSU算法自动计算
    ret, binary_image = cv2.threshold(denoised_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 计算每个像素点与右侧一个像素点的灰度差值绝对值，并统计每一列像素的差值和
    height, width = binary_image.shape[:2]
    column_sums = np.zeros(width)  # 存储每一列像素的差值和
    row_sums = np.zeros(height)  # 存储每一行像素的差值和
    for y in range(height):
        for x in range(width - 1):
            diff = abs(int(binary_image[y, x]) - int(binary_image[y, x + 1]))  # 计算灰度差值绝对值
            column_sums[x] += diff  # 累加差值到相应列

    # 计算最右侧像素点与0的差值绝对值，并添加到column_sums的最后一列
    column_sums[-1] = abs(int(binary_image[0, -1]) - 0)

    for y in range(height - 1):
        for x in range(width):
            diff = abs(int(binary_image[y, x]) - int(binary_image[y + 1, x]))  # 计算灰度差值绝对值
            row_sums[y] += diff  # 累加差值到相应行

    # 计算最下方像素点与0的差值绝对值，并添加到row_sums的最后一行
    row_sums[-1] = abs(int(binary_image[-1, 0]) - 0)

    find_peak_list_x = find_peak(column_sums)
    find_peak_list_y = find_peak(row_sums)

    # 由DataMatrix二维码的码制可知，二维码的大小都是偶数个模块，因此分割线的数量应该为奇数。
    # 比较水平方向和竖直方向的分割线数量，如果不相等，则图中噪声较大，需要增大滤波带宽，直到两者相等或者带宽大于30
    # 带宽若是大于30，边缘信息会因为滤波而丢失，无法进行解码
    while len(find_peak_list_x) != len(find_peak_list_y) or len(find_peak_list_x) % 2 == 0 or len(find_peak_list_y) % 2 == 0 and denoised_bw < 30:
        # 增加带宽
        denoised_bw += 2
        denoised_image = cv2.bilateralFilter(image, denoised_bw, 75, 75)

        # 采用cv2进行二值化，二值化的阈值用OTSU算法自动计算
        ret, binary_image = cv2.threshold(denoised_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 计算每个像素点与右侧一个像素点的灰度差值绝对值，并统计每一列像素的差值和
        height, width = binary_image.shape[:2]
        column_sums = np.zeros(width)  # 存储每一列像素的差值和
        row_sums = np.zeros(height)  # 存储每一行像素的差值和
        for y in range(height):
            for x in range(width - 1):
                diff = abs(int(binary_image[y, x]) - int(binary_image[y, x + 1]))  # 计算灰度差值绝对值
                column_sums[x] += diff  # 累加差值到相应列

        # 计算最右侧像素点与0的差值绝对值，并添加到column_sums的最后一列
        column_sums[-1] = abs(int(binary_image[0, -1]) - 0)

        for y in range(height - 1):
            for x in range(width):
                diff = abs(int(binary_image[y, x]) - int(binary_image[y + 1, x]))  # 计算灰度差值绝对值
                row_sums[y] += diff  # 累加差值到相应行

        # 计算最下方像素点与0的差值绝对值，并添加到row_sums的最后一行
        row_sums[-1] = abs(int(binary_image[-1, 0]) - 0)

        find_peak_list_x = find_peak(column_sums)
        find_peak_list_y = find_peak(row_sums)

    if len(find_peak_list_x) != len(find_peak_list_y):
        print("二维码噪声过大，无法进行解码")
        return None

    # 得到网格划分后每个网格的四个顶点坐标
    grid_points_list = grid_division(peak_list_x=find_peak_list_x, peak_list_y=find_peak_list_y, is_point=False)

    # 得到网格列表内每个网格的灰度分布趋势
    grid_grey_tend_list = grid_iterative_segmentation(grid_points_list_input=grid_points_list,
                                                      binary_image_input=binary_image)
    # 得到每个网格二值化后的值，1表示黑色，0表示白色
    predict_grey_value_list = predict_gray_value(grid_grey_tend_list)

    # 将预测值转换为二维数组。其中，二维数组的大小是predict_grey_value_list的平方根，因为二维码是一个n阶矩阵
    matrix_ranks = int(np.sqrt(len(predict_grey_value_list)))
    dmtx_matrix = np.array(predict_grey_value_list).reshape(matrix_ranks, matrix_ranks)

    # 返回解码后的数据矩阵
    return dmtx_matrix



root = tk.Tk()

root.withdraw()

f_path = filedialog.askopenfilename()

test_data = decode_dmtx(f_path)

# 设置灰度图的大小
length = int(test_data.size ** 0.5)

# 将灰度值列表转成灰度图像
predict_grey_value_array = np.array(test_data, dtype=np.uint8)
predict_grey_image = predict_grey_value_array.reshape(length, length)

# 缩放灰度图
resized_grey_image = cv2.resize(predict_grey_image, (400, 400), interpolation=cv2.INTER_AREA)

# 在图像四周添加50像素的空白
resized_grey_image = cv2.copyMakeBorder(resized_grey_image, 50, 50, 50, 50, cv2.BORDER_CONSTANT, value=0)

# 保存路径
save_path_dir = os.path.join(os.path.dirname(f_path), 'output')
os.makedirs(save_path_dir, exist_ok=True)
save_path = os.path.join(save_path_dir, os.path.basename(f_path))

# 保存图片
cv2.imwrite(save_path, resized_grey_image)

# 解码灰度图
decoded_data = dmtx.decode(cv2.imread(save_path, cv2.IMREAD_GRAYSCALE))

# 如果解码失败，将图片边缘进行反色处理后再次解码
if not decoded_data:
    # 在图像四周添加50像素的空白
    resized_grey_image = cv2.copyMakeBorder(resized_grey_image, 50, 50, 50, 50, cv2.BORDER_CONSTANT, value=255)

    # 保存路径
    save_path_dir = os.path.join(os.path.dirname(f_path), 'output')
    os.makedirs(save_path_dir, exist_ok=True)
    save_path = os.path.join(save_path_dir, os.path.basename(f_path))

    # 保存图片
    cv2.imwrite(save_path, resized_grey_image)

    # 解码灰度图
    decoded_data = dmtx.decode(cv2.imread(save_path, cv2.IMREAD_GRAYSCALE))

# 打印解码结果
print(decoded_data)
