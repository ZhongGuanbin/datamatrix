import tkinter as tk
from tkinter import filedialog

import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def decode_dmtx(image_path):
    # 采用cv2读取路径为image_path的PNG图片，将其灰度化
    image = cv2.imread(image_path)
    image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_LINEAR)

    # CLAHE对比度增强
    clahe = cv2.createCLAHE(clipLimit=1, tileGridSize=(5, 5))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = clahe.apply(image)

    # 采用cv2的双边滤波方法，去噪,去除高斯白噪声，尽可能保留边缘信息
    denoised_image = cv2.bilateralFilter(image, 7, 75, 75)

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

    # 计算水平方向分割线
    # 去基线，y＜7000的认为是噪音，扣除
    # 7000的取值原因：最大尺寸26*26，所以1格灰度值1000/26*255
    # 至少有1格模式边，就算数据区某行或者某列全为0，再加上部分变形，至少7000
    y_sub_h = np.where(column_sums < 2400, 0, column_sums)
    y_sub_v = np.where(row_sums < 2400, 0, row_sums)

    # 对y进行差分
    y_diff_h = np.diff(y_sub_h)
    y_diff_v = np.diff(y_sub_v)

    # 对y_diff求符号
    y_diff_sign_h = np.sign(y_diff_h)
    y_diff_sign_v = np.sign(y_diff_v)

    # 对y_diff_sign再求一次差分
    y_diff_sign_diff_h = np.diff(y_diff_sign_h)

    # plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定中文字体，例如黑体、SimHei等
    # plt.rcParams['axes.unicode_minus'] = False  # 解决负号'-'显示为方块的问题
    fig, ax = plt.subplots(2, 1, figsize=(12, 9), gridspec_kw={'hspace': 0.5})

    ax[0].plot(np.array(column_sums), color='black', linewidth=2)
    ax[0].set_xlim([0, len(np.array(column_sums))])
    ax[0].set_ylim([min(np.array(column_sums)), 1.2 * max(np.array(column_sums))])
    ax[0].set_xlabel('Horizontal position(px)', fontsize=12)
    ax[0].set_ylabel('Grayscale difference', fontsize=12)
    ax[0].set_title('Horizontal gray distribution map', fontsize=16)
    ax[0].legend(['Gray distribution'], loc='upper right', fontsize=10)
    ax[0].tick_params(labelsize=10)

    ax[1].plot(np.array(row_sums), color='black', linewidth=2)
    ax[1].set_xlim([0, len(np.array(row_sums))])
    ax[1].set_ylim([min(np.array(row_sums)), 1.2 * max(np.array(row_sums))])
    ax[1].set_xlabel('Vertical position(px)', fontsize=12)
    ax[1].set_ylabel('Grayscale difference', fontsize=12)
    ax[1].set_title('Vertical gray distribution map', fontsize=16)
    ax[1].legend(['Gray distribution'], loc='upper right', fontsize=10)
    ax[1].tick_params(labelsize=10)

    plt.show()
    fig.savefig('p3.png', dpi=500)

    # 遍历y_diff_sign_diff，
    # 将其中y_diff_sign_diff[i]=-2(单峰)或y_diff_sign_diff[i] == -1（梯形峰，有多个相等的最大值）的i值变为i+1后存储到数组peak_list_origin中
    peak_list_origin = []
    for i in range(len(y_diff_sign_diff_h)):
        if y_diff_sign_diff_h[i] == -2 or y_diff_sign_diff_h[i] == -1:
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

        # 因为单个二维码水平或者竖直方向最多有26个模块，图像在1000*1000的空间内，所以正常情况下的最小间距是1000/26大约40，
        # 再结合我们预测的真实峰间距，若峰间距小于max（8，prediction_peak_spacing / 2），则认为是同一组的峰，
        # 直到峰间距大于max（8，prediction_peak_spacing / 2）
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
                peak_y_list.append(y_sub_h[int(peak_list_group[i][j])])
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

    # # 统计每一行像素的差值和，最下方的像素点则与0做差值
    # gy = np.abs(np.diff(np.asarray(resized_binary_image), axis=0))
    # row_sum = np.sum(gy, axis=1)
    # row_sum = np.append(row_sum, 0)
    #
    # # 垂直方向横纵坐标数组
    # y_v = row_sum
    #
    # plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定中文字体，例如黑体、SimHei等
    # plt.rcParams['axes.unicode_minus'] = False  # 解决负号'-'显示为方块的问题
    # fig, ax = plt.subplots(figsize=(16, 10))
    # ax.plot(np.array(y_v), color='black', linewidth=2)
    # ax.set_xlim([0, len(np.array(y_v))])
    # ax.set_ylim([min(np.array(y_v)), 1.2 * max(np.array(y_v))])
    # ax.set_xlabel('竖直方向位置 （px）', fontsize=22)
    # ax.set_ylabel('灰度差值', fontsize=22)
    # ax.set_title('竖直方向灰度分布图', fontsize=26)
    # plt.legend(['灰度分布'], loc='upper right', fontsize=18)
    # ax.tick_params(labelsize=18)
    # plt.rcParams.update({'font.size': 24})
    # plt.show()
    # fig.savefig('vertical_gray_distribution10.png', dpi=500)

    # # 计算垂直方向分割线
    # find_peak_list_y = find_peak(y_v)
    #
    # # 得到网格划分后每个网格的四个顶点坐标
    # grid_points_list = grid_division(peak_list_x=find_peak_list_x, peak_list_y=find_peak_list_y, is_point=False)
    #
    # # 得到网格列表内每个网格的灰度分布趋势
    # grid_grey_tend_list = grid_iterative_segmentation(grid_points_list_input=grid_points_list,
    #                                                   binary_image_input=resized_binary_image)
    # # 得到每个网格二值化后的值，1表示黑色，0表示白色
    # predict_grey_value_list = predict_gray_value(grid_grey_tend_list)
    #
    # # 返回解码后的数据矩阵
    # return predict_grey_value_list


root = tk.Tk()

root.withdraw()

f_path = filedialog.askopenfilename()

decode_dmtx(f_path)
