import cv2
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde


# 定义寻峰函数
def find_peak(y_input):
    # 去基线，y＜4000的认为是噪音，扣除
    # 4000的取值原因：最大尺寸26*26，所以1格灰度值400/26*255
    # 至少有1格模式边，就算数据区某行或者某列全为0，所以至少4000
    y_sub = np.where(y_input < 4000, 0, y_input)

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

    # 对peak_list_diff_origin排序
    peak_list_diff_origin.sort()

    # 将peak_list_diff_origin数据转换为pandas序列
    peak_list_diff = pd.Series(peak_list_diff_origin)

    # 计算指数加权移动平均
    peak_list_diff_ewma = peak_list_diff.ewm(alpha=0.3).mean()

    # 得到预测间距值
    prediction_peak_spacing = peak_list_diff_ewma.iloc[-1]
    print(prediction_peak_spacing)

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

        # 因为单个二维码水平或者竖直方向最多有26个模块，图像在400*400的空间内，所以正常情况下的最小间距是400/26=15，
        # 再结合我们预测的真实峰间距，若峰间距小于max（5，prediction_peak_spacing/2），则认为是同一组的峰，
        # 直到峰间距大于max（5，prediction_peak_spacing/2）
        if x_diff <= max(5, prediction_peak_spacing / 2):
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

    return peak_list_group_max


# 采用cv2读取路径为D:\datamatrix\2393的文件夹下有一张名为input.png的PNG图片，将其灰度化
image = cv2.imread(r"D:\datamatrix\standard.png", cv2.IMREAD_GRAYSCALE)

# 采用cv2进行灰度均衡处理
equalized_image = cv2.equalizeHist(image)

# 采用cv2的双边滤波方法，去噪
denoised_image = cv2.bilateralFilter(equalized_image, 9, 75, 75)

# 采用cv2进行二值化，二值化的阈值用OTSU算法自动计算
ret, binary_image = cv2.threshold(denoised_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# 缩放到400*400的像素空间中
resized_binary_image = cv2.resize(binary_image, (400, 400), interpolation=cv2.INTER_LINEAR)

# 计算每个像素点与右侧一个像素点的灰度差值
Gx = np.abs(np.diff(np.asarray(resized_binary_image), axis=1))

# 统计每一列像素的差值和，最右侧的像素点则与0做差值
col_sum = np.sum(Gx, axis=0)
col_sum = np.append(col_sum, 0)

# 横纵坐标数组
x = np.arange(len(col_sum))
x = np.linspace(0, len(x) - 1, 400)
y = col_sum

# 寻峰
peak_list_find_peak = find_peak(y)
print(peak_list_find_peak)


# # 遍历peak_x_list_find_peak_group，奇数分割线往左，偶数分割线往右
# peak_x_list_find_peak_group_lr = []
# for i in range(len(peak_x_list_find_peak_group)):
#
#     # 如果只有一个峰，则直接将其添加到peak_x_list_find_peak_group_lr中
#     if len(peak_x_list_find_peak_group[i]) == 1:
#         peak_x_list_find_peak_group_lr.append(peak_x_list_find_peak_group[i][0])
#
#     # 如果有多个峰，则判断是奇数还是偶数，奇数选【0】，偶数选【-1】
#     else:
#         if i % 2 == 0:
#             peak_x_list_find_peak_group_lr.append(peak_x_list_find_peak_group[i][0])
#         else:
#             peak_x_list_find_peak_group_lr.append(peak_x_list_find_peak_group[i][-1])

# # 剔除小于7的间隔值
# peak_x_diff = peak_list_diff_origin[peak_list_diff_origin > 7]
#
# # 给予7到15的间隔值很小的权重
# weights = np.ones_like(peak_x_diff)
# weights[(peak_x_diff <= 15)] = 0.0001
#
# # 估算概率密度函数
# kde = gaussian_kde(peak_x_diff, weights=weights)
# x = np.linspace(peak_x_diff.min(), peak_x_diff.max(), 100)
# y_input = kde(x)
#
# # 计算概率密度函数的最大值
# idx = np.argmax(y_input)
# most_likely_value = x[idx]
# print(peak_x_diff, most_likely_value)
